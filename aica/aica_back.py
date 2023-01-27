# pip install -q openai
# pip install gradio -q

# I want to create a card for mom. I love her so much. She likes to cook for me. She is the best cook I know.
# Generate a few birthday card titles that is suitable for this story:

import gradio as gr
import random
from PIL import Image
import requests
from io import BytesIO
import sys

import json

with open('message_suggestions.json', encoding='utf8') as f:
    card_type_to_message = json.load(f)

with open('title_suggestions.json', encoding='utf8') as f:
    card_type_to_title = json.load(f)

im1 = "ai1.png"
im2 = "charlie.png"
im3 = "christmas.png"
im4 = "girl2.png"

# user choices
card_types = list(card_type_to_message.keys())

num_card_variations = 4
num_title_variations = 4

# title_suggestions = ["1. Happy birthday!", "2. Coolest birthday!", "3. A year younger"]
prompt_examples = ["a photo of a cat",
                   "a drawing of a mom cooking",
                   "a drawing of a birthday cake",
                   "your idea",
                   "go back to choose a new card type"]

story_example = "Give me more information about you and the receiver of the card" \
                " and I will suggest a few suitable card titles." \
                " Example1: 'I want to create a card for my mom. I love her so much." \
                " She likes to cook for me. She is the best cook I know.'" \
                " Example2: 'I want a birthday card for my mom. She likes dogs.'" \
                " -------- If you do not want to write a story, just [1], and we will " \
                "generate a few titles based on your card type:)"

good_bye_message = "Great. We're happy you're satisfied with the card." \
                   " Now you can download it and send it as e-card by email or print it out."
yes_no_options = ["yes", "no"]
title_options = ["use AICA suggestions", "you idea"]
message_options = title_options
next_steps_after_picking_best_card = ["I'm done", "I want to modify the card", "I want to add a title and a message"]


def enum_list(input_list):
    return [str(i) + ". " + str(item) for i, item in enumerate(input_list, 1)]


response_after_card_generated = f"Great! Thank you. We have generated 4 cards for you to choose. " \
                                f"Let's take a look at them on the right pane."
gpt3 = "text-curie-001"  # "text-davinci-003" #"text-curie-001",

intents = {
    "greeting": f"Hello! I am AICA, your AI card generator. What kind of card would you like to generate? "
                f"{enum_list(card_types)}. Please type a digit (without a dot).", }

greeting = f"Hello! I am AICA, your AI card generator. What kind of card would you like to generate? " \
           f"{enum_list(card_types)}. Please type a digit (without a dot)."

history = []
bot_actions = []

card_spec = {}


def add_message_to_card(card, message):
    # only output one option
    return im2


def add_title(card, seed, title):
    output = [("ai1.png", 1), ("charlie.png", 2), ("christmas.png", 3), ("girl2.png", 4)]
    random.shuffle(output)
    return output


def suggest_title(story, card_type):
    return card_type_to_title[card_type]


def suggest_message(story, card_type):
    return card_type_to_message[card_type]


def suggest_message_using_gpt3(story, card_type):
    return card_type_to_message[card_type]


def reformulate_message(message, length, tone, style):
    # use gpt3 to do grammar correct and reformulating the message
    # e.g. make it shorter, have a particular tone, style
    return message


def generate_cards(prompt):
    return [("ai1.png", 1), ("charlie.png", 2), ("christmas.png", 3), ("girl2.png", 4)]


def modify(prompt, input_image, seed):
    # use different scheduler
    # first image is the same
    return [(input_image, seed), ("charlie.png", 1), ("christmas.png", 1), ("girl2.png", 1)]


# cards = gr.State([])


def chatbot(message, cards):
    response = ""
    print(bot_actions)
    # title_suggestions = []
    # message_suggestions = []
    next_steps = enum_list(next_steps_after_picking_best_card)
    print(cards)

    if not bot_actions or bot_actions[-1] == "final":
        response = greeting
        bot_actions.append("greeting")

    elif bot_actions[-1] == "greeting":
        options = range(1, len(card_types) + 1)
        if message in [str(num) for num in options]:

            card_spec["type"] = card_types[int(message) - 1]
            response += f" You picked '{card_spec['type']} card'."
            response += f"  Next, what image should we draw on your card? Here are some options: " \
                        f"{enum_list(prompt_examples)}"
            bot_actions.append("choose_prompt")

        else:
            response = f"Please choose a number from: {list(options)}"

    elif bot_actions[-1] == "choose_prompt":
        options = list(range(1, len(prompt_examples) + 1))
        if message in [str(num) for num in options]:
            if message == "4":
                response = f"Please give me a prompt. E.g. 'a drawing of a cake on a green table' "
                bot_actions.append("user_gave_manual_prompt")
            elif message == "5":
                response = greeting
                bot_actions.append("greeting")
            else:
                response = response_after_card_generated
                card_spec["user_prompt"] = prompt_examples[int(message) - 1]
                cards = generate_cards(prompt_examples[int(message) - 1])
                bot_actions.append("return_cards")
        else:
            response = f"Please choose a number from: {list(options)}"

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

            response = f"Good job! Now let's choose the next step: {next_steps} "
            bot_actions.append("card_image_done")
        else:
            response = f"Choose the card you like the most. Type a digit from {list(options)}"

    elif bot_actions[-1] == "card_image_done":
        options = range(1, len(next_steps) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                response = good_bye_message
                bot_actions.append("final")
            elif message == "2":
                response = "You want to modify the card? Sure, then please give us a new prompt " \
                           "that contain description about things you want to change. " \
                           "E.g. old prompt = 'A drawing of a cat', " \
                           ">> new prompt = 'A watercolor drawing of a white cat'"
                bot_actions.append("modify_card")
            elif message == "3":
                response = f"What should be the title of you card? {enum_list(title_options)}"
                bot_actions.append("add_title")
        else:
            response = f"Please choose a number from: {next_steps}"

    elif bot_actions[-1] == "modify_card":
        new_prompt = message
        card_spec["user_prompt"] = new_prompt
        old_seed = cards[-1][1]
        cards = modify(new_prompt, cards[-1][0], old_seed)
        response = "Here is your new cards."
        bot_actions.append("return_cards")

    elif bot_actions[-1] == "add_title":
        options = range(1, len(title_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                response = story_example
                bot_actions.append("user_gave_story")
            elif message == "2":
                response = "Tell me what we should put as a title."
                bot_actions.append("user_gave_title")

    elif bot_actions[-1] == "user_gave_story":
        # use story to suggest
        if message == "1":

            # todo if user type 1, use AICA suggestion, else use gpt3
            title_suggestions = suggest_title(message, card_spec["type"]) + ["your idea"]
        else:  # todo this should be from gpt3
            title_suggestions = suggest_title(message, card_spec["type"]) + ["your idea"]
            card_spec["story"] = message
        card_spec["title_suggestions"] = title_suggestions
        response = f"Here are our suggestions for the title, please pick a number: {enum_list(title_suggestions)}"
        bot_actions.append("user_chose_title")

    elif bot_actions[-1] == "user_chose_title":
        title_suggestions = card_spec["title_suggestions"]
        options = range(1, len(title_suggestions) + 1)
        if message in [str(num) for num in options]:
            print("user_chose_title message", message)
            if message == str(len(title_suggestions)):  # user picked "your idea"
                response = "Tell me what we should put as a title."
                bot_actions.append("user_gave_title")
            else:
                # put title (message) to the card, generate new card
                seed = cards[-1][1]
                card = cards[-1][0]
                title = title_suggestions[int(message) - 1]
                # card_with_title = (add_title(card, title), seed)  # todo
                cards = add_title(card, seed, title)  # todo
                bot_actions.append("return_card_with_title")
                # response = f"Do you want to put a message below your card? {enum_list(yes_no_options)}"
                # bot_actions.append("return_card_with_title")
                # bot_actions.append("title_done")
        else:
            response = f"Please choose a number from: {enum_list(title_suggestions)}"

    elif bot_actions[-1] == "user_gave_title":
        seed = cards[-1][-1]
        title = message
        card = cards[-1][0]
        # card_with_title = (add_title(card, title), seed)  # todo
        cards = add_title(card, seed, title)  # generate_cards("") #[card_with_title]
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
            response = f"Do you want to put a message below your card? {enum_list(yes_no_options)}"

            bot_actions.append("title_done")
        else:
            response = f"Choose the card you like the most. Choose a number from: {list(options)}"


    elif bot_actions[-1] == "title_done":
        options = range(1, len(yes_no_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                response = f"How do you want to generate the message? {enum_list(message_options)}"
                bot_actions.append("choose_message_options")
            else:
                response = "Then, we are finished!"
                bot_actions.append("final")
        else:
            response = f"Do you want to put a message below your card? {enum_list(yes_no_options)}"

    elif bot_actions[-1] == "choose_message_options":
        options = range(1, len(message_options) + 1)
        if message in [str(num) for num in options]:
            if message == "1":
                if "story" not in card_spec:
                    response = story_example
                else:
                    response = "You've already told us about your story, so we will generate suggestions based on " \
                               "this story." \
                               " Please type any key to proceed."
                bot_actions.append("user_gave_story_for_message")
            else:
                response = "Tell us what we should write under the card."
                bot_actions.append("user_chose_message")
        else:
            response = f"How do you want to generate the message? {enum_list(message_options)}"

    elif bot_actions[-1] == "user_gave_story_for_message":

        if "story" not in card_spec:
            card_spec["story"] = message

        message_suggestions = suggest_message(card_spec["story"], card_spec["type"])  # + ["your idea"]
        card_spec["message_suggestions"] = message_suggestions
        response = f"Please choose a number from: {enum_list(message_suggestions)}"
        bot_actions.append("user_chose_message")

    elif bot_actions[-1] == "user_chose_message":
        message_suggestions = card_spec["message_suggestions"]
        options = range(1, len(message_suggestions) + 1)
        if message in [str(num) for num in options]:
            card_message = message_suggestions[int(message) - 1]
            current_card = cards[-1][0]
            new_image = add_message_to_card(current_card, card_message)  # [(im4, 1)]
            seed = cards[-1][-1]
            cards = [(new_image, seed)]
            response = "We are finished. Enjoy. Thank you for using AICA!"
            bot_actions.append("final")
        else:

            response = f"Please choose a number from: {enum_list(message_suggestions)}"

    if bot_actions[-1] == "return_cards":
        card_options = list(range(1, num_card_variations + 1))
        response += f" Choose the card you like the most: {card_options} "
        bot_actions.append("picked_best_card")

    if bot_actions[-1] == "return_card_with_title":
        card_options = list(range(1, num_title_variations + 1))
        response += f" Choose the card you like the most: {card_options} "
        bot_actions.append("picked_best_card_with_title")

    if not response:
        response = greeting
        bot_actions.append("greeting")

    history.append((message, response))
    return history, cards, str(cards)


def image_path_to_image(cards):
    # cards = [(image, seed)..]
    img_list = []
    for card, seed in cards:
        # response = requests.get(card)
        # img = Image.open(BytesIO(response.content))
        img = Image.open(card)
        img_list.append(img)
    return img_list


with gr.Blocks() as demo:
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

            card = gr.Image(None).style(height=10, rounded=False)
            out_text = gr.Textbox(placeholder="What is your name?", visible=True)
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
