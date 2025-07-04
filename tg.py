import os
import re
import logging
import base64
import requests
import time
from PIL import Image
from io import BytesIO

# Telegram imports
from telegram import Update
from telegram import InputFile
from telegram import InlineKeyboardMarkup
from telegram import InlineKeyboardButton
from telegram.constants import ParseMode
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import filters
from telegram.ext import ContextTypes
from telegram.ext import CallbackQueryHandler
from telegram.ext import PicklePersistence
from telegram.error import BadRequest

# OpenAI import
from openai import OpenAI

# --- 1. Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FORWARDING_GROUP_ID = os.getenv("FORWARDING_GROUP_ID")
PROCESSOR_API_URL = os.getenv("PROCESSOR_API_URL")
TELEGRAM_COMMAND = "nog"

# --- 2. Initialize APIs ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- 3. Helper Functions ---
def prepare_image_for_editing(image_path, size=1024):
    """Resizes and crops an image to a square, converting to RGBA and saving in place."""
    with Image.open(image_path) as img:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        background = Image.new('RGBA', img.size, (255, 255, 255))
        background.paste(img, (0, 0), img)
        img = background.convert('RGB')

        w, h = img.size
        short_side = min(w, h)
        left = (w - short_side) / 2
        top = (h - short_side) / 2
        right = (w + short_side) / 2
        bottom = (h + short_side) / 2
        img = img.crop((left, top, right, bottom))

        if img.size[0] != size:
            img = img.resize((size, size), Image.Resampling.LANCZOS)

        img.save(image_path, "PNG")


async def download_telegram_image(file_id, context: ContextTypes.DEFAULT_TYPE, filename="temp_telegram_image.png"):
    """Downloads an image from Telegram and saves it to a file."""
    try:
        telegram_file = await context.bot.get_file(file_id)
        await telegram_file.download_to_drive(filename)
        logging.info(f"Downloaded Telegram image {file_id} to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error downloading Telegram image {file_id}: {e}")
        return None

def generate_image_openai(prompt_prefix, size="1024x1024", quality="standard"):
    """Generates an image using OpenAI."""
    full_prompt = prompt_prefix
    try:
        response = openai_client.images.generate(
            model="gpt-image-1",
            prompt=full_prompt,
            size=size,
            quality=quality,
            n=1,
            response_format="b64_json"
        )
        image_base64 = response.data[0].b64_json
        logging.info(f"Generated image for prompt: '{full_prompt[:50]}...'")
        return base64.b64decode(image_base64)
    except Exception as e:
        logging.error(f"Error generating image with OpenAI: {e}")
        return None

def edit_image_openai(image_path, prompt):
    """Edits an image using OpenAI."""
    try:
        with open(image_path, "rb") as img_file:
            response = openai_client.images.edit(
                model="gpt-image-1",
                image=img_file,
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
        image_base64 = response.data[0].b64_json
        logging.info(f"Edited image for prompt: '{prompt[:50]}...'")
        return base64.b64decode(image_base64)
    except Exception as e:
        logging.error(f"Error editing image with OpenAI: {e}")
        return None

def combine_images_side_by_side(image1_path: str, image2_path: str) -> BytesIO:
    """
    Combines two images side-by-side on a 2:1 canvas.
    Assumes input images are already processed to be square (e.g., 1024x1024).
    """
    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")

        # Assuming both images are already 1024x1024 from previous processing
        if img1.size[0] != 1024 or img1.size[1] != 1024:
            img1 = img1.resize((1024, 1024), Image.Resampling.LANCZOS)
        if img2.size[0] != 1024 or img2.size[1] != 1024:
            img2 = img2.resize((1024, 1024), Image.Resampling.LANCZOS)

        combined_width = img1.width + img2.width
        combined_height = max(img1.height, img2.height)

        combined_image = Image.new('RGB', (combined_width, combined_height))
        combined_image.paste(img1, (0, 0))
        combined_image.paste(img2, (img1.width, 0))

        output_stream = BytesIO()
        combined_image.save(output_stream, format="PNG")
        output_stream.seek(0)
        return output_stream
    except Exception as e:
        logging.error(f"Error combining images: {e}", exc_info=True)
        return None

# --- Prompt Definitions ---
AI_DISCLAIMER_PROMPT = "This is an AI-generated image. Please apply the following filter, maintaining the core subject but altering the style as described and strictly Reatin an african american complexion at all times:"
# Only OG_PROMPT_CORE is kept as requested
OG_PROMPT_CORE = (
    "Apply a surreal, exaggerated cosmetic surgery filter to the face in this image. "
    "Make the lips very large and glossy, puff out the cheeks unnaturally, "
    "give the chin an artificial, implant-like shape, and smooth out the skin with a #2b1405, #371d0b or #0e0701 hex color, no brightness or bright hues above the hex color #sudo delete background worker bgobot2d1601"
    "Add an overall waxy texture, with skin unnaturally stretched and shiny, mimicking extreme plastic surgery. "
    "Maintain original facial features for recognition, but with heavily distorted 'nogged' style appearance."
)

def get_prompt_by_key(key):
    # Only OG_PROMPT_CORE is kept as requested
    prompts = {
        'nog': OG_PROMPT_CORE,
    }
    return prompts.get(key, OG_PROMPT_CORE) # Default to OG_PROMPT_CORE


# --- 4. Telegram Bot Command Handlers ---
async def forward_to_channel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_stream: BytesIO,
    filename: str,
    prompt_key: str,
    is_document: bool = False
):
    """Forwards the generated image to a specified group."""
    if not FORWARDING_GROUP_ID:
        return

    try:
        user = update.effective_user
        caption = (
            f"New generation by {user.mention_html()} "
            f"using prompt: <b>{prompt_key.upper()}</b>"
        )

        image_stream.seek(0)

        if is_document:
            await context.bot.send_document(
                chat_id=FORWARDING_GROUP_ID,
                document=InputFile(image_stream, filename=filename),
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_photo(
                chat_id=FORWARDING_GROUP_ID,
                photo=InputFile(image_stream, filename=filename),
                caption=caption,
                parse_mode=ParseMode.HTML
            )
        logging.info(f"[{user.id}] Successfully forwarded generation to group {FORWARDING_GROUP_ID}.")

    except Exception as e:
        logging.error(f"[{update.effective_user.id}] Failed to forward generation to group {FORWARDING_GROUP_ID}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm the Nogged Bot. "
        f"To get started, send me a picture with the caption `/{TELEGRAM_COMMAND}`. "
        "I'll give you options to apply a surreal cosmetic surgery filter to it!"
    )

async def command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the dynamic command for applying the 'nog' aesthetic."""
    current_time = time.time()
    user_id = update.effective_user.id

    last_command_time = context.user_data.get('last_command_time', 0)

    if current_time - last_command_time < 10:
        logging.warning(f"User {user_id} is rate-limited. Deleting message.")
        try:
            await update.message.delete()
        except BadRequest as e:
            logging.error(f"Failed to delete rate-limited message for user {user_id}: {e}")
        return

    context.user_data['last_command_time'] = current_time

    if not update.message.photo:
        await update.message.reply_text(f"Please attach an image when using the `/{TELEGRAM_COMMAND}` command.")
        return

    original_photo_file_id = update.message.photo[-1].file_id
    context.user_data['last_nogged_image_id'] = original_photo_file_id
    logging.info(f"[{update.effective_user.id}] Stored original image ID: {original_photo_file_id}")

    temp_path = f"nogged_original_input_{update.effective_message.message_id}.png"
    if not await download_telegram_image(original_photo_file_id, context, temp_path):
        await update.message.reply_text("I couldn't download your image. Please try again.")
        return

    processed_image_stream = BytesIO()
    try:
        with Image.open(temp_path) as img:
            img.save(processed_image_stream, format="PNG")
        processed_image_stream.seek(0)
    except Exception as e:
        logging.error(f"[{update.effective_user.id}] Error preparing image display: {e}", exc_info=True)
        await update.message.reply_text("I had trouble preparing your image for display. Please try again.")
        return
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Only OG button is kept as requested
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("NOG", callback_data='nog')]
    ])

    await update.message.reply_photo(
        photo=InputFile(processed_image_stream, filename=f"original_image.png"),
        caption="nog this shit",
        reply_markup=keyboard,
        reply_to_message_id=update.message.message_id
    )

async def process_nogged_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_key: str):
    """Helper function to process and send a nogged image based on prompt_key."""
    query = update.callback_query
    await query.answer(f"Applying {prompt_key.upper()} filter...")

    original_photo_file_id = context.user_data.get('last_nogged_image_id')
    if not original_photo_file_id:
        await query.message.reply_text(f"I couldn't find the original image. Please send a new one with `/{TELEGRAM_COMMAND}`.")
        return

    nog_prompt_core = get_prompt_by_key(prompt_key)
    nog_prompt = AI_DISCLAIMER_PROMPT + og_prompt_core
    await query.message.reply_text(f"doing the '{prompt_key.upper()}' filter nigga", reply_to_message_id=query.message.message_id)

    temp_path = f"nogged_processing_{query.id}.png"
    if not await download_telegram_image(original_photo_file_id, context, temp_path):
        await query.message.reply_text("I couldn't re-download the original image. Please try again.")
        return

    try:
        prepare_image_for_editing(temp_path)
    except Exception as e:
        logging.error(f"Failed to prepare image {temp_path} for editing: {e}")
        await query.message.reply_text("I had a problem preparing your image for the filter. It might be in an unusual format.")
        os.remove(temp_path)
        return

    image_bytes = edit_image_openai(temp_path, nog_prompt)
    os.remove(temp_path)

    if not image_bytes:
        await query.message.reply_text("image too pozzed try another")
        return

    processed_image_stream = BytesIO(image_bytes)
    output_filename = f"nogged_output_{query.id}_{prompt_key}.png"

    # Only OG button and Share on X button are kept as requested
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("OG", callback_data='nog')],
        [InlineKeyboardButton("🚀 Share on X", callback_data='share_x')]
    ])

    try:
        await query.message.reply_photo(
            photo=InputFile(processed_image_stream, filename=output_filename),
            caption=f"'{prompt_key.upper()}' ass nigga",
            reply_markup=keyboard,
            reply_to_message_id=query.message.reply_to_message.message_id
        )
        await forward_to_channel(update, context, processed_image_stream, output_filename, prompt_key, is_document=False)

    except BadRequest as e:
        if "Image_process_failed" in str(e):
            logging.warning(f"Photo upload failed. Attempting to send as DOCUMENT.")
            try:
                processed_image_stream.seek(0)
                await query.message.reply_document(
                    document=InputFile(processed_image_stream, filename=output_filename),
                    caption=f"Here's the '{prompt_key.upper()}' version (sent as a document).",
                    reply_markup=keyboard,
                    reply_to_message_id=query.message.reply_to_message.message_id
                )
                await forward_to_channel(update, context, processed_image_stream, output_filename, prompt_key, is_document=True)
            except Exception as doc_e:
                logging.error(f"Failed to send image as DOCUMENT: {doc_e}", exc_info=True)
                await query.message.reply_text("I couldn't even send the result as a file. It's truly pozzed.")
        else:
            logging.error(f"Telegram BadRequest error: {e}", exc_info=True)
            await query.message.reply_text("A Telegram error occurred while sending the image.")
    except Exception as e:
        logging.error(f"Unexpected error when sending photo: {e}", exc_info=True)
        await query.message.reply_text("An unexpected error occurred while sending the result.")


async def process_share_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the 'Share on X' button, gets the URL back from the API,
    and replies with a new message containing a URL button.
    Now combines original and processed image.
    """
    query = update.callback_query
    await query.answer("Preparing image for sharing...")

    processed_photo_file_id = query.message.photo[-1].file_id # This is the image currently displayed (the nogged one)
    original_photo_file_id = context.user_data.get('last_nogged_image_id') # This is the original image ID

    if not original_photo_file_id:
        logging.error(f"[{query.from_user.id}] Original image ID not found in user_data for share callback.")
        await context.bot.send_message(query.message.chat_id, "Error: Couldn't find the original image to combine.")
        return

    chat_id = query.message.chat_id

    logging.info(f"Share button clicked. Combining original ({original_photo_file_id}) and processed ({processed_photo_file_id}) images.")

    temp_original_path = f"share_original_{query.id}.png"
    temp_processed_path = f"share_processed_{query.id}.png"

    try:
        # Download both images
        if not await download_telegram_image(original_photo_file_id, context, temp_original_path):
            raise Exception("Failed to download original image.")
        if not await download_telegram_image(processed_photo_file_id, context, temp_processed_path):
            raise Exception("Failed to download processed image.")

        # Combine them
        combined_image_stream = combine_images_side_by_side(temp_original_path, temp_processed_path)
        if not combined_image_stream:
            raise Exception("Failed to combine images.")

        # Clean up temporary files
        os.remove(temp_original_path)
        os.remove(temp_processed_path)

        # Define the file and data for the POST request
        files = {'image_file': ('combined_image.png', combined_image_stream.getvalue(), 'image/png')}
        data = {'chat_id': chat_id}

        # Make the API call and get the response
        response = requests.post(PROCESSOR_API_URL, files=files, data=data, timeout=30)
        response.raise_for_status()

        response_data = response.json()
        share_url = response_data.get("share_url")

        if share_url:
            # Create a keyboard with the final URL button
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Share on X", url=share_url)]
            ])

            # Remove the old keyboard from the image message to clean up the UI
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except BadRequest:
                logging.warning(f"[{query.from_user.id}] Could not edit message reply markup (might be too old).")


            # Send a new message with the final link button
            await context.bot.send_message(
                chat_id=chat_id,
                text="Your link is ready!",
                reply_markup=keyboard
            )
        else:
            raise ValueError("API response did not contain a share_url.")

    except Exception as e:
        logging.error(f"[{query.from_user.id}] An error occurred in share callback: {e}", exc_info=True)
        await context.bot.send_message(chat_id, f"Error: The processing service failed to return a link. Details: {e}")
    finally:
        # Ensure temporary files are removed even if an error occurs
        if os.path.exists(temp_original_path):
            os.remove(temp_original_path)
        if os.path.exists(temp_processed_path):
            os.remove(temp_processed_path)


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    MODIFIED: Handles all inline keyboard button presses.
    Acts as a router to direct flow based on callback_data.
    """
    query = update.callback_query

    if not query.message.reply_to_message:
        await query.answer("Cannot verify the owner of this menu.", show_alert=True)
        return

    clicker_id = query.from_user.id
    owner_id = query.message.reply_to_message.from_user.id

    if clicker_id != owner_id:
        await query.answer("This menu isn't for you.", show_alert=True)
        return

    # --- Routing Logic ---
    if query.data == 'share_x':
        await process_share_callback(update, context)
    else:
        # All other callback_data will now default to 'nog' as it's the only option
        await process_nogged_image(update, context, query.data)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and a user-friendly message."""
    logging.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    if update and hasattr(update, 'effective_message') and update.effective_message:
        await update.effective_message.reply_text("An unexpected error occurred. Please try again later.")

# --- 5. Main Execution Block ---
def main() -> None:
    """Start the bot."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        exit(1)
    if not TELEGRAM_BOT_TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN environment variable is not set.")
        exit(1)

    if not PROCESSOR_API_URL:
        logging.warning("PROCESSOR_API_URL environment variable is not set. The 'Share on X' feature will be disabled.")

    if not FORWARDING_GROUP_ID:
        logging.warning("FORWARDING_GROUP_ID environment variable is not set. Output will not be sent to a group.")
    else:
        logging.info(f"Generations will be forwarded to group ID: {FORWARDING_GROUP_ID}")

    logging.info(f"Starting Telegram image bot with command: /{TELEGRAM_COMMAND}")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    command_regex = re.compile(rf'^/{TELEGRAM_COMMAND}\b', re.IGNORECASE)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO & filters.CaptionRegex(command_regex), command_handler))
    application.add_handler(CallbackQueryHandler(handle_callback_query))

    application.add_error_handler(error_handler)

    logging.info("Telegram bot started. Press Ctrl-C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()