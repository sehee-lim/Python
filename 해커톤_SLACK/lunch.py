import os
import boto3
import random
# from flask import Flask, request
from slack_sdk import WebClient
from slackeventsapi import SlackEventAdapter
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

lunch = ["파스타", "피자", "치킨", "김밥", "족발", "순두부찌개",
         "김치찌개", "된장찌개", "떡볶이", "제육볶음", "돈까스",
         "찜닭", "닭볶음탕", "햄버거", "스시", "수육국밥", "삼겹살",
         "쌀국수", "냉면", "김치볶음밥", "부대찌개", "콩나물국밥",
         "라멘", "비빔밥", "짜장면", "짬뽕", "칼국수", "삼계탕",
         "쌈밥", "메밀소바", "카레", "우동", "간장게장", "육회",
         "만두", "설렁탕", "팟타이", "탕수육", "마라탕", "마라샹궈",
         "닭발", "곱창", "막창", "대창", "등갈비", "보쌈", "조개구이",
         "양꼬치", "돼지갈비", "소갈비", "오리고기", "대하", "백숙",
         "간장새우", "사케동", "밥버거", "회", "샤브샤브", "스테이크",
         "전", "죽", "갈비찜", "갈비탕", "청국장", "추어탕", "월남쌈",
         "백반", "깐풍기", "고추장찌개", "잔치국수", "샌드위치", "샐러드",
         "베이글", "시래기국", "미역국", "닭강정", "육개장", "라면",
         "텐동", "쭈꾸미", "온면", "감자탕", "뼈해장국", "주먹밥"]

channel_id = "C057ETZJR5K"
SLACK_BOT_TOKEN = "xoxb-5256167969204-5249288940070-xFlfaQl65wcofkygCCWBLwF2"
SLACK_APP_TOKEN = "xapp-1-A057HRWNDS6-5256777839635-e98f48764760778a1c4f6fb12b2f58ee3084a904dc1cc0c23340663058928611"
SLACK_SIGNING_SECRET = "720b87dee12d12531197c00794a6437e"

app = App(
    # signing_secret = SLACK_SIGNING_SECRET
    token = SLACK_BOT_TOKEN
)

slack_client = WebClient(token = SLACK_BOT_TOKEN)

@app.event("message")
@app.event("app_mention")
def regex(event_data, message, say):
    # say(f"Hello <@{message['user']}>")
    user = message["user"]
    # text = message["text"]
    random_int = random.randint(0, len(lunch))
    say(f"<@{user}>! 점심 메뉴를 추천해드릴게요.", lunch[random_int])
    # say(f"Hi <@{user}>! You said: {text}")
    # message = event_data["event"]
    # channel = message["channel"]
    
        
if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()