# pip install -q openai
# pip install gradio -q
# pip install -q git+https://github.com/huggingface/diffusers
# pip install -q git+https://github.com/huggingface/transformers
# pip install -q pynvml
# pip install -q diffusers["torch"]
# pip install accelerate

# I want to create a card for mom. I love her so much. She likes to cook for me. She is the best cook I know.
# Generate a few birthday card titles that is suitable for this story:

import gradio as gr
import random
from PIL import Image
import requests
from io import BytesIO
import sys
from PIL import Image, ImageOps, ImageFont, ImageDraw
import textwrap
import json
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, \
    EulerDiscreteScheduler


im1 = "im1.jpg"
im2 = "im2.jpg"
im3 = "im3.jpg"
im4 = "im4.jpg"
im_paths = [im1, im2, im3, im4]

def generate_cards(prompt, num_images=4, scheduler="EulerDiscrete", num_steps=20):

    l = [(Image.open(card),1) for card in im_paths]
    return l


def modify(prompt, input_image, seed, num_steps=20):
    l = [(Image.open(card), seed) for card in im_paths[1:]]
    return [(input_image, seed)] + l


#####################################################

with open('fonts.json', encoding='utf8') as f:
    font_dict = json.load(f)

title_fonts = font_dict["title_fonts"]
msg_fonts = font_dict["msg_fonts"]


def add_border_title(img, title, font, font_size=50, color="white"):
    """takes a chosen img and returns img with border and title"""

    # left, top, right, bottom
    border = (39, 39, 39, 156)
    image = Image.open(img)
    image_border = ImageOps.expand(image, border=border, fill=color)

    req = requests.get(font)
    title_font = ImageFont.truetype(BytesIO(req.content), font_size)

    # center text horizontally
    font_w, font_h = title_font.getsize(title)
    title_x = (image_border.size[0] - font_w) / 2
    title_y = 80
    image_draw = ImageDraw.Draw(image_border)

    # draw text multiple times to create shadow
    image_draw.text((title_x - 1, title_y), title, fill="gray", font=title_font)
    image_draw.text((title_x + 1, title_y), title, fill=color, font=title_font)
    image_draw.text((title_x, title_y - 1), title, fill="gray", font=title_font)
    image_draw.text((title_x, title_y + 1), title, fill=color, font=title_font)

    return image_border


def add_message(img, recipient, sender, message, msg_font=msg_fonts[0], msg_font_size=15, msg_color="black"):
    """takes a chosen image with title and returns img with title and message"""

    msg_req = requests.get(msg_font)
    msg_font = ImageFont.truetype(BytesIO(msg_req.content), msg_font_size)
    # img = add_border_title(img, "Happy Birthday", title_fonts[0])
    # img = Image.open(img)

    img_draw = ImageDraw.Draw(img)
    # left, top, right, bottom
    # border = (39, 40, 39, 156)

    # recipient
    recipient_x = 60
    recipient_y = img.size[1] - 135

    # apply text wrap to prevent bleed
    msg_list = textwrap.wrap(message, 70)
    msg_new = '\n'.join(msg_list)

    space = " " * 4
    recipient_msg = "Dear " + recipient + ",\n" + space + msg_new
    img_draw.text((recipient_x, recipient_y), recipient_msg, fill=msg_color, font=msg_font)

    # sender
    sender_x = img.size[0] - 100
    sender_y = img.size[1] - 50
    img_draw.text((sender_x, sender_y), sender, fill=msg_color, font=msg_font)

    return img


def get_cards_with_diff_title_styles(input_card, title, seed, fonts=title_fonts):
    outputs = []
    for f in fonts:
        img = add_border_title(input_card, title, f)
        outputs.append((img, seed))
    return outputs


# out = add_message(img="diffusion1.png", recipient="mom", sender="me", message="Blaaaaa")
#
# out.show()
if False:
    o = get_cards_with_diff_title_styles(input_card="diffusion1.png", title="Happy Birthday my mom Happy Birth", seed=2)
    print(o)
    for i, seed in o:
        i.show()
    sys.exit()
# ######################## chatbot config ######################

with open('message_suggestions.json', encoding='utf8') as f:
    card_type_to_message = json.load(f)

with open('title_suggestions.json', encoding='utf8') as f:
    card_type_to_title = json.load(f)



# user choices
card_types = list(card_type_to_message.keys())

num_card_variations = 4
num_title_variations = 4

prompt_options = ["a photo of a cat",
                   "a drawing of a mom cooking",
                   "a drawing of a birthday cake",
                   "your idea",
                   "go back to choose a new card type"]

prompt_examples = ["a drawing of a cake on a green table",
                    "a photo of a cat",
                    "a drawing of a birthday cake",
                    "a watercolor drawing of a mom cooking"]

story_example = " Example1: 'I want to create a card for my mom. I love her so much." \
                " She likes to cook for me. She is the best cook I know.'\n" \
                " Example2: 'I want a birthday card for my mom. She likes dogs.'\n\n" \
                " Or type 1 to skip."

good_bye_message = "Great. We're happy you're satisfied with the card." \
                   " Now you can download it and send it as e-card by email or print it out."
yes_no_options = ["yes", "no"]
title_options = ["use AICA suggestions", "you idea"]
message_options = title_options
next_steps_after_picking_best_card = ["I'm done", "I want to modify the card", "I want to add a title and a message"]


def enum_list(input_list):
    return [str(i) + ". " + str(item) for i, item in enumerate(input_list, 1)]


response_after_card_generated = f"Great! Thank you. We have generated 4 cards for you. " \
                                f"Let's take a look at them on the right pane."

greeting = f"Hello! I am AICA, your AI card generator. What kind of card would you like to generate? " \
           f"You can see the options on the right side. To choose your option, type a number (without a dot)."

history = []
bot_actions = []
card_spec = {}
# ######################## gpt3 ######################


gpt3 = "text-davinci-003"  # "text-davinci-003", "text-curie-001",
import openai

openai.api_key = "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"


# possible parameters: https://beta.openai.com/docs/api-reference/completions/create
def suggest_title(story, card_type, tone="funny", use_gpt3=False):
    if use_gpt3:
        model = "text-davinci-003"
        max_tokens = 30
        num_texts = 3
        if story:
            prompt = f"Generate a very short {tone} {card_type} card slogan suitable for this story(in one line):"
            prompt = "story:[" + story + "] " + prompt
        else:
            prompt = f"Generate a very short {tone} {card_type} card slogan:"
        print("prompt:", prompt)
        completion = openai.Completion.create(
            engine=model,
            prompt=prompt,
            n=num_texts,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        results = [completion.choices[i].text.strip() for i in range(num_texts)]

        return results

    return card_type_to_title[card_type]


# r = suggest_title("", "graduation", tone="funny", use_gpt3=True)
# print(r)

def suggest_message(story, card_type, tone="funny", use_gpt3=False):
    if use_gpt3:
        model = "text-davinci-003"
        max_tokens = 50
        num_texts = 3
        if story:
            prompt = f"Generate a {tone} message for {card_type} card suitable for this story:"
            prompt = "story:[" + story + "] " + prompt
        else:
            prompt = f"Generate a {tone} message for {card_type} card:"
        print("prompt:", prompt)
        completion = openai.Completion.create(
            engine=model,
            prompt=prompt,
            n=num_texts,
            temperature=0.5,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        results = [completion.choices[i].text.strip() for i in range(num_texts)]

        return results
    return card_type_to_message[card_type]


# r = suggest_message("I want to make a card for my boyfriend. He studied medicine.", "graduation", tone="loving",
# use_gpt3=True)
# print(r)


def reformulate_message(message, length="short", tone="funny", style="informal"):
    # use gpt3 to do grammar correct and reformulating the message
    # e.g. make it shorter, have a particular tone, style

    model = "text-curie-001"
    max_tokens = 100
    num_texts = 1
    prompt = f"{message}\n Reformulate this text. Make it {length} {tone} {style}:"
    print("prompt:", prompt)
    completion = openai.Completion.create(
        engine=model,
        prompt=prompt,
        n=num_texts,
        temperature=0.5,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    # results = [completion.choices[i].text.strip() for i in range(num_texts)]

    return completion.choices[0].text


# message = "I so proud of you, doktor! You've worked so hard and accomplished so much. I can't wait to see what the
# future has in store for you. Congratulations on your graduation! I love you"
# r = reformulate_message(message, length="short", tone="funny", style="informal")
# print(r)


def add_message_to_card(card, message):
    # only output one option
    return im2


def add_title(card, seed, title):
    output = [(im1, 1), (im2, 2), (im3, 3), (im4, 4)]
    random.shuffle(output)
    return output


message_tone = "funny"
choose_the_best_card = f" Choose the card you like the most by typing a number."
recipient, sender = "Min", "Nana"


# ######################## gradio gui ######################

def chatbot(message, cards):
    response = ""
    print(bot_actions)
    # title_suggestions = []
    # message_suggestions = []
    next_steps = enum_list(next_steps_after_picking_best_card)
    print(cards)

    user_choices = ""

    if not bot_actions or bot_actions[-1] == "final":
        response = greeting
        bot_actions.append("greeting")
        user_choices = enum_list(card_types)

    elif bot_actions[-1] == "greeting":
        options = range(1, len(card_types) + 1)
        if message in [str(num) for num in options]:

            card_spec["type"] = card_types[int(message) - 1]
            response += f" You picked '{card_spec['type']} card'."
            response += f"  Next, what image should we draw on your card?" \
                        f" We've listed a few suggestions for you." \
                        f" Please see the right pane and choose a number."
            bot_actions.append("choose_prompt")
            user_choices = enum_list(prompt_options)

        else:
            response = f"Please choose a number from the right pane."
            user_choices = enum_list(card_types)


    elif bot_actions[-1] == "choose_prompt":
        options = list(range(1, len(prompt_options) + 1))
        if message in [str(num) for num in options]:
            if message == "4":
                response = f"Please give me a prompt. See examples on the right pane."
                user_choices = prompt_examples
                bot_actions.append("user_gave_manual_prompt")
            elif message == "5":
                response = greeting
                bot_actions.append("greeting")
                user_choices = enum_list(card_types)
            else:
                response = response_after_card_generated
                card_spec["user_prompt"] = prompt_options[int(message) - 1]
                cards = generate_cards(prompt_options[int(message) - 1])
                bot_actions.append("return_cards")
        else:
            response = f"Please choose a number from the right pane."
            user_choices = enum_list(prompt_options)

    elif bot_actions[-1] == "user_gave_manual_prompt":
        card_spec["user_prompt"] = message
        response = response_after_card_generated
        cards = generate_cards(message)
        bot_actions.append("return_cards")

    elif bot_actions[-1] == "picked_best_card":
        options = range(1, num_card_variations + 1)
        # next_steps = ["1. I'm finished", "2. modify the card", "3. add a title and a message"]
        if message in [str(num) for num in options]:
            card_id = int(message) - 1
            cards = [cards[card_id]]

            response = f"Good job! Now let's choose the next step."
            user_choices = next_steps  # already enum
            bot_actions.append("card_image_done")
        else:
            response = choose_the_best_card
            user_choices = list(options)

    elif bot_actions[-1] == "card_image_done":
        options = range(1, len(next_steps) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                response = good_bye_message
                bot_actions.append("final")
            elif message == "2":
                response = "You want to modify the card? Sure, then please give us a new prompt " \
                           "that contain description about things you want to change. "
                user_choices = [f"You current prompt is {card_spec['user_prompt']}",
                                "------------------",
                                "Example:", "old prompt = 'A drawing of a cat'",
                                "new prompt = 'A watercolor drawing of a white cat'"]
                bot_actions.append("modify_card")
            elif message == "3":
                # response = f"What should be the title of you card? {enum_list(title_options)}"
                response = f"What should be the title of you card?"
                user_choices = enum_list(title_options)
                bot_actions.append("add_title")
        else:
            response = f"Please choose a number listed in the option box."
            user_choices = next_steps

    elif bot_actions[-1] == "modify_card":
        new_prompt = message
        card_spec["user_prompt"] = new_prompt
        old_seed = cards[-1][1]
        cards = modify(new_prompt, cards[-1][0], old_seed)
        response = "Done! We've generated a few card variations for you."
        bot_actions.append("return_cards")

    elif bot_actions[-1] == "add_title":
        options = range(1, len(title_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":

                response = "Give me more information about you and the receiver of the card" \
                           " and I will suggest a few suitable card titles." \
                           " If you do not want to write a story, just type 1, and we will " \
                           "generate a few titles based on your card type:)"
                user_choices = [story_example]
                bot_actions.append("user_gave_story")
            elif message == "2":
                response = "Tell me what we should put as a title."
                bot_actions.append("user_gave_title")
        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(title_options)

    elif bot_actions[-1] == "user_gave_story":
        # use story to suggest
        if message == "1":  # use AICa suggestion
            title_suggestions = suggest_title(message, card_spec["type"]) + ["your idea"]

        else:  # todo this should be from gpt3
            title_suggestions = suggest_title(message, card_spec["type"], tone=message_tone, use_gpt3=True) + [
                "your idea"]
            card_spec["story"] = message
        card_spec["title_suggestions"] = title_suggestions
        response = f"Here are our suggestions for the title. Please pick a number."
        user_choices = enum_list(title_suggestions)
        bot_actions.append("user_chose_title")

    elif bot_actions[-1] == "user_chose_title":
        title_suggestions = card_spec["title_suggestions"]
        options = range(1, len(title_suggestions) + 1)
        if message in [str(num) for num in options]:
            if message == str(len(title_suggestions)):  # user picked "your idea"
                response = "Tell me what we should put as the title."
                bot_actions.append("user_gave_title")
            else:
                # put title (message) to the card, generate new card
                seed = cards[-1][1]
                card = cards[-1][0]
                title = title_suggestions[int(message) - 1]
                # card_with_title = (add_title(card, title), seed)  # todo
                cards = get_cards_with_diff_title_styles(card, title, seed)  # todo
                bot_actions.append("return_card_with_title")
                # response = f"Do you want to put a message below your card? {enum_list(yes_no_options)}"
                # bot_actions.append("return_card_with_title")
                # bot_actions.append("title_done")
        else:
            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(title_suggestions)

    elif bot_actions[-1] == "user_gave_title":
        seed = cards[-1][-1]
        title = message
        card = cards[-1][0]
        # card_with_title = (add_title(card, title), seed)  # todo
        cards = get_cards_with_diff_title_styles(card, title, seed)  # generate_cards("") #[card_with_title]
        bot_actions.append("return_card_with_title")
        # bot_actions.append("title_done")
        # response = f"Do you want to put a message below your card? {enum_list(yes_no_options)}"


    elif bot_actions[-1] == "picked_best_card_with_title":
        options = range(1, num_title_variations + 1)
        # next_steps = ["1. I'm finished", "2. modify the card", "3. add a title and a message"]
        if message in [str(num) for num in options]:
            card_id = int(message) - 1
            cards = [cards[card_id]]

            # response = f"Good job! Now let's choose the next step: {next_steps} "
            response = f"Do you want to put a message below your card?"

            bot_actions.append("title_done")
            user_choices = enum_list(yes_no_options)
        else:
            response = choose_the_best_card
            user_choices = list(options)


    elif bot_actions[-1] == "title_done":
        options = range(1, len(yes_no_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                # response = f"How do you want to generate the message? {enum_list(message_options)}"
                response = f"How do you want to generate the message? Please choose a number."
                user_choices = enum_list(message_options)
                bot_actions.append("choose_message_options")
            else:
                response = "Then, we are finished! Thank you for using AICA:)"
                bot_actions.append("final")
        else:
            response = f"Do you want to put a message below your card?"
            user_choices = enum_list(yes_no_options)

    elif bot_actions[-1] == "choose_message_options":
        options = range(1, len(message_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                if "story" not in card_spec:
                    response = "Give me more information about you and the receiver of the card" \
                               " and I will suggest a personalized message for you card." \
                               " If you do not want to write a story, just type [1], and we will " \
                               "generate suggestions based on your card type:)"
                    user_choices = [story_example]
                else:
                    response = "You've already told us about your story, so we will generate suggestions based on " \
                               "this story." \
                               " Please type any key to proceed."
                bot_actions.append("user_gave_story_for_message")
            else:
                response = "Tell us what we should write under the card."
                bot_actions.append("user_chose_message")
        else:
            response = f"How do you want to generate the message? Choose a number."
            user_choices = enum_list(message_options)

    elif bot_actions[-1] == "user_gave_story_for_message":

        if "story" not in card_spec:
            card_spec["story"] = message

        message_suggestions = suggest_message(card_spec["story"], card_spec["type"])  # + ["your idea"]
        card_spec["message_suggestions"] = message_suggestions
        response = f"Here are our suggestions for the card message. Please choose a number."
        user_choices = enum_list(message_suggestions)
        bot_actions.append("user_chose_message")

    elif bot_actions[-1] == "user_chose_message":
        message_suggestions = card_spec["message_suggestions"]
        options = range(1, len(message_suggestions) + 1)
        if message in [str(num) for num in options]:
            card_message = message_suggestions[int(message) - 1]
            current_card = cards[-1][0]
            new_image = add_message(current_card, recipient, sender, msg_fonts[0], card_message)  # [(im4, 1)]
            seed = cards[-1][-1]
            cards = [(new_image, seed)]
            response = "We are finished! Enjoy. Thank you for using AICA:)"
            bot_actions.append("final")
        else:

            response = f"Please choose a number listed in the option box."
            user_choices = enum_list(message_suggestions)

    if bot_actions[-1] == "return_cards":
        card_options = list(range(1, num_card_variations + 1))
        response += choose_the_best_card
        bot_actions.append("picked_best_card")
        user_choices = card_options

    if bot_actions[-1] == "return_card_with_title":
        card_options = list(range(1, num_title_variations + 1))
        response += choose_the_best_card
        bot_actions.append("picked_best_card_with_title")
        user_choices = card_options

    if not response:
        response = greeting
        bot_actions.append("greeting")
        user_choices = enum_list(card_types)

    history.append((message, response))
    return history, cards, "\n".join([str(i) for i in user_choices])


def image_path_to_image(cards):
    # cards = [(image, seed)..]
    img_list = []
    for card, seed in cards:
        # response = requests.get(card)
        # img = Image.open(BytesIO(response.content))
        if isinstance(card, str):
            card= Image.open(card)

        img_list.append(card)
    return img_list


with gr.Blocks(css=".gradio-container {font-size: 20}") as demo:
    cards = gr.State([])

    with gr.Row() as row:
        with gr.Column():
            gr.Markdown(
                """
                # AICA
                Tell AICA to generate a personalized card for you!
                """)

            # https://gradio.app/docs/#chatbot
            display1 = gr.Chatbot()  # gr.outputs.Chatbot()#.style(color_map=("green", "pink")) #[("", intents[
            # "greeting"])]
            text1 = gr.Textbox(label="You", lines=1)  # gr.inputs.Textbox(label="You", lines=2)
            button1 = gr.Button(label="Send", show_label=True)

        with gr.Column():
            gr.Markdown(
                """
                _______
                # Your card
                """)

            card = gr.Image(None).style(height=10)
            out_text = gr.Textbox(placeholder="Options", visible=True, lines=5, max_lines=100)
            # current_card = gr.Image(im_path,label="Your card", shape=(100, None))
            gallery = gr.Gallery(None,
                                 label="Generated images", show_label=False, elem_id="gallery"
                                 ).style(grid=2, height="100")

        # text1.change(chatbot1,text1, display1)

        button1.click(chatbot, inputs=[text1, cards], outputs=[display1, cards,
                                                               out_text])  # out_text should be im path from
        # generated text, not object...
        # out_text.change(add_image, inputs=out_text, outputs=gallery)
        # out_text.change(url_to_images, inputs=out_text, outputs=gallery)
        # out_text.change(lambda :images, outputs=gallery)
        out_text.change(image_path_to_image, inputs=cards, outputs=gallery)

demo.launch(debug=True)
