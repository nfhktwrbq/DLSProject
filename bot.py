from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from config import TOKEN
import aiohttp
import requests
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher.filters import Text
from aiogram.dispatcher import FSMContext
import os
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from styleTransfer import StyleTransfer
import torchvision.transforms as transforms
import asyncio
from GAN.pix2pix import Pix2pix

SOCK5IP = 'ss-01.s5.ynvv.cc'
SOCK5PORT = '999'
SOCK5LOGIN = '273764704'
SOCK5PASS = 'lMP3XQqd'

BUTTON1_NAME = 'Style Transfer'
BUTTON2_NAME = 'GAN'

WEBHOOK_HOST = 'https://sleepy-plateau-32377.herokuapp.com/'
WEBHOOK_PATH = TOKEN
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# webserver settings
WEBAPP_HOST = 'localhost'  # or ip
WEBAPP_PORT = int(os.environ.get('PORT', 5000))

button_styleTransfer = KeyboardButton(BUTTON1_NAME)
button_GAN = KeyboardButton(BUTTON2_NAME)
greet_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(button_styleTransfer).add(button_GAN)

PROXY_URL = 'socks5://' + SOCK5IP + ':' + SOCK5PORT
PROXY_AUTH = aiohttp.BasicAuth(login=SOCK5LOGIN, password=SOCK5PASS)

if SOCK5IP is None:
    bot = Bot(token=TOKEN, proxy=PROXY_URL)
else:
    bot = Bot(token=TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)
storage = MemoryStorage()

dp = Dispatcher(bot, storage=storage)

def savePhoto(photo, filename, path):
    photo_url = "https://api.telegram.org/file/bot{0}/{1}".format(
        TOKEN, photo.file_path)
    r = requests.get(photo_url, proxies=dict(http='socks5://' + SOCK5LOGIN + ':' + SOCK5PASS + '@' + SOCK5IP + ':' + SOCK5PORT,
                                 https='socks5://' + SOCK5LOGIN + ':' + SOCK5PASS + '@' + SOCK5IP + ':' + SOCK5PORT))
    if r.status_code == 200:
        with open(path + '/' + filename, 'wb') as f:
            f.write(r.content)
    else:
        return


def createConvDir(convType):
    parentDir = 'Conversions'
    dirPref = 'conv_'
    path = "./"

    lilstDir = [x for x in os.listdir('.')]
    if parentDir not in lilstDir:
        os.mkdir(path + parentDir)

    lilstDir = [x for x in os.listdir(path + parentDir)]
    l = [int(y) for y in [x.replace(dirPref+convType, '') for x in lilstDir] if y.isdigit()]
    if len(l) == 0:
        indmax = 1
    else:
        indmax = max(l) + 1
    resPath = path + parentDir + '/' + dirPref+convType + str(indmax)
    os.mkdir(resPath)
    return resPath

# States
class Form(StatesGroup):
    select = State()
    ST1 = State()
    ST2 = State()
    GAN = State()

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await Form.select.set()
    await message.reply("Hi!\nChoose conversion", reply_markup=greet_kb)

async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(dp):
    await bot.delete_webhook()
    await dp.storage.close()
    await dp.storage.wait_closed()

@dp.message_handler(state='*', commands='cancel')
@dp.message_handler(Text(equals='cancel', ignore_case=True), state='*')
async def cancel_handler(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        return

    await message.reply("Cancel", reply_markup=greet_kb)
    await state.finish()
    await message.reply('Cancelled.', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(lambda message: message.text not in [BUTTON1_NAME, BUTTON2_NAME], state=Form.select)
async def process_gender_invalid(message: types.Message):
    return await message.reply("Bad conversion name. Choose your conversion from the keyboard.")

@dp.message_handler(state=Form.select)
async def process_name(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['select'] = message.text
    if message.text == BUTTON1_NAME:
        await Form.ST1.set()
        await message.reply("Load style image", reply_markup=types.ReplyKeyboardRemove())
    else:
        await Form.GAN.set()
        await message.reply("Load guitar scetch", reply_markup=types.ReplyKeyboardRemove())

@dp.message_handler(content_types=['photo'], state=Form.ST1)
async def answer_photo(msg: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['ST1'] = createConvDir('ST')
    await Form.ST2.set()
    photo = await bot.get_file(msg.photo[-1].file_id)
    savePhoto(photo, 'style', data['ST1'])
    await bot.send_message(msg.from_user.id, 'Ok. Load content image')


@dp.message_handler(content_types=['photo'], state=Form.ST2)
async def answer_photo(msg: types.Message, state: FSMContext):
    photo = await bot.get_file(msg.photo[-1].file_id)
    data = await state.get_data()

    savePhoto(photo, 'content', data['ST1'])
    resPath = data['ST1'] + '/' + 'res.jpg'

    await bot.send_message(msg.from_user.id, 'Wait...')

    out = StyleTransfer(data['ST1'] + '/' + 'style', data['ST1'] + '/' + 'content').getOutput()
    img = transforms.ToPILImage(mode='RGB')(out.cpu()[0])
    img.save(resPath)

    await asyncio.sleep(1)
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    media.attach_photo(types.InputFile(resPath), 'Ready!')
    await msg.answer_media_group(media=media)

    await state.finish()

@dp.message_handler(content_types=['photo'], state=Form.GAN)
async def answer_photo(msg: types.Message, state: FSMContext):
    path = createConvDir('GAN')
    photo = await bot.get_file(msg.photo[-1].file_id)
    savePhoto(photo, 'input', path)

    await bot.send_message(msg.from_user.id, 'Wait...')

    p2p = Pix2pix('', '')
    p2p.loadModels('./GAN/weights/')
    out = p2p.getOutput(path + '/' + 'input')
    resPath = path + '/' + 'output.jpg'

    img = transforms.ToPILImage(mode='RGB')(out.cpu())
    img.save(resPath)

    await asyncio.sleep(1)
    await types.ChatActions.upload_photo()
    media = types.MediaGroup()
    media.attach_photo(types.InputFile(resPath), 'Ready!')
    await msg.answer_media_group(media=media)

    await state.finish()

if __name__ == '__main__':
    #executor.start_polling(dp)
    executor.start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )