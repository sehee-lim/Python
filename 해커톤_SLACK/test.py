import os
import boto3
# from flask import Flask, request
from slack_sdk import WebClient
from slackeventsapi import SlackEventAdapter
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

channel_id = "C057ETZJR5K"
SLACK_BOT_TOKEN = "xoxb-5256167969204-5249288940070-xFlfaQl65wcofkygCCWBLwF2"
SLACK_APP_TOKEN = "xapp-1-A057HRWNDS6-5256777839635-e98f48764760778a1c4f6fb12b2f58ee3084a904dc1cc0c23340663058928611"
SLACK_SIGNING_SECRET = "720b87dee12d12531197c00794a6437e"

# dynamodb = boto3.resource('dynamodb', region_name = 'ap-northeast-2')
# table_name = "slack_info"
# table = dynamodb.Table(table_name)

app = App(
    # signing_secret = SLACK_SIGNING_SECRET
    token = SLACK_BOT_TOKEN
)

# app = Flask(__name__)
slack_client = WebClient(token = SLACK_BOT_TOKEN)
# slack_events_adapter = SlackEventAdapter(SLACK_SIGNING_SECRET, "/slack/events", app)

@app.event("message")
@app.event("app_mention")
def regex(event_data, message, say):
    # say(f"Hello <@{message['user']}>")
    user = message["user"]
    text = message["text"]
    say(f"Hi <@{user}>! You said: {text}")
    # message = event_data["event"]
    # channel = message["channel"]
    print(f"{user}: {text}")
    # table.put_item(Item = {"user": user, "text": text})
    # print(message["text"])
    



# @slack_events_adapter.on("message")
# @app.event("message")
# def handle_message(event_data, message):
#     message = event_data["event"]
#     channel = message["channel"]
#     user = message["user"]
#     text = message["text"]
#     print(f"Message from {user}: {text}")
#     slack_client.chat_postMessage(channel=channel, text=f"Hi <@{user}>! You said: {text}")
    
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
    # app.run(port = 3000)



# class SlackAPI:
#     def __init__(self, token):
#         self.client = WebClient(token)
        
#     def post_message(self, channel_id):
#         result = self.client.chat_postMessage(
#             channel = channel_id,
#             text = "hello"
#         )
        
#         return result
    
#     def post_thread_message(self, channel_id, message_ts):
#         result = self.client.chat_postMessage(
#             channel=channel_id,
#             text = text,
#             thread_ts = message_ts
#         )
#         return result

# slack = SlackAPI(SLACK_APP_TOKEN)
# slack.post_message(channel_id)

