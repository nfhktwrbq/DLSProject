from aiogram import Bot, filters, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from config import TOKEN
import aiohttp
import requests
import uuid
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton

SOCK5IP = 'ss-01.s5.ynvv.cc'
SOCK5PORT = '999'
SOCK5LOGIN = '273764704'
SOCK5PASS = 'lMP3XQqd'

BUTTON1_NAME = 'Style Transfer'
BUTTON2_NAME = 'GAN'

button_styleTransfer = KeyboardButton(BUTTON1_NAME)
button_GAN = KeyboardButton(BUTTON2_NAME)
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(button_styleTransfer).add(button_GAN)

PROXY_URL = 'socks5://' + SOCK5IP + ':' + SOCK5PORT  #ss-01.s5.ynvv.cc:999'
PROXY_AUTH = aiohttp.BasicAuth(login=SOCK5LOGIN, password=SOCK5PASS)

if SOCK5IP is None:
    bot = Bot(token=TOKEN, proxy=PROXY_URL)
else:
    bot = Bot(token=TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)

dp = Dispatcher(bot)

"""@dp.message_handler(filters.CommandStart())
async def send_welcome(message: types.Message):
    # So... At first I want to send something like this:
    await message.reply("Do you want to see many pussies? Are you ready?")

    # Wait a little...
    await asyncio.sleep(1)

    # Good bots should send chat actions...
    await types.ChatActions.upload_photo()

    # Create media group
    media = types.MediaGroup()

    # Attach local file
    media.attach_photo(types.InputFile('orig.webp'), 'Cat!')
    # More local files and more cats!
    media.attach_photo(types.InputFile('style.jpg'), 'More cats!')

    # You can also use URL's
    # For example: get random puss:
    media.attach_photo('http://lorempixel.com/400/200/cats/', 'Random cat.')

    # And you can also use file ID:
    # media.attach_photo('<file_id>', 'cat-cat-cat.')

    # Done! Send media group
    await message.reply_media_group(media=media)"""

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет!\nВыбери преобразование", reply_markup=greet_kb)

"""@dp.message_handler(commands=['start', 'help'])
async def process_help_command(message: types.Message):
    await message.reply("Напиши мне что-нибудь, и я отпрпавлю этот текст тебе в ответ!")"""

@dp.message_handler()
async def echo_message(msg: types.Message):
    if msg.text is BUTTON1_NAME:
        await bot.send_message(msg.from_user.id, 'Отправь два фото квадратной формы.\n'
                                                 'Первое стиль второе контент.' , reply_markup=ReplyKeyboardRemove())




@dp.message_handler(content_types=['photo'])
async def answer_photo(msg: types.Message):
    print(msg.photo[-1].file_id)
    photo = await bot.get_file(msg.photo[-1].file_id)
    photo_url = "https://api.telegram.org/file/bot{0}/{1}".format(
        TOKEN, photo.file_path)
    print(photo_url)
    #proxies = {'socks5': 'socks5://273764704:lMP3XQqd@ss-01.s5.ynvv.cc:999'}
    proxies = {
        'http': '45.77.222.251:3128',
        'https': '45.77.222.251:3128'

    }
    r = requests.get(photo_url, proxies=dict(http='socks5://273764704:lMP3XQqd@ss-01.s5.ynvv.cc:999',
                                 https='socks5://273764704:lMP3XQqd@ss-01.s5.ynvv.cc:999'))
    file_name = str(uuid.uuid4()) + '.png'
    if r.status_code == 200:
        with open('./' + file_name, 'wb') as f:
            f.write(r.content)
    else:
        bot.reply_to(msg, 'something fails...')
        return

    """document_id = msg.document.file_id
    file_info = await bot.get_file(document_id)
    fi = file_info.file_path
    name = msg.document.file_name
    urllib.request.urlretrieve(f'https://api.telegram.org/file/bot{TOKEN}/{fi}',f'./{name}')"""
    await bot.send_message(msg.from_user.id, 'Файл успешно сохранён')

if __name__ == '__main__':
    executor.start_polling(dp)