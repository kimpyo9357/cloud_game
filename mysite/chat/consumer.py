from channels.generic.websocket import AsyncWebsocketConsumer
import json
from .game import run
import asyncio
import ctypes

before_client = 0
count = 0
client_list = {}
play_game = False

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        global client_list        
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'chat_%s' % self.room_name
        self.client = ":".join(map(str,self.scope['client']))
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        message = self.client +' join in this channels\n if you want to play game,\n you should input text \n"play game random/greedy/maximin/DQN/friends"'

        # "room" 그룹에 메시지 전송
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'client' : self.client,
                'group' : self.room_name
            }
        )
        if (self.room_name not in client_list.keys()):
            client_list[self.room_name] = []
        

        await self.accept()

    async def disconnect(self, close_code):
        global client_list
        message = self.client +' join out this channels'
        if len(client_list[self.room_name]) == 1:
            del client_list[self.room_name]
            play_game = False

        # "room" 그룹에 메시지 전송
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'client' : self.client,
                'group' : self.room_name
            }
        )
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        global before_client, client_list
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        if (play_game):
            with open('action.txt','w') as f:
                f.write(message)
        paring_message = message.split()
        if (before_client != self.client):
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': message,
                    'group' : self.room_name
                }
            )
            #before_client = self.client

    async def chat_message(self,event):
        global client_list, play_game, human      
        message = event['message']
        temp = message.split()
        if (len(temp) == 5):
            if(temp[1] == 'join' and temp[2] == 'in' and not(event['client'] in client_list[event['group']])):
                client_list[event['group']].append(event['client'])
            elif(temp[1] == 'join' and temp[2] == 'out' and(event['client'] in client_list[event['group']])):
                client_list[event['group']].pop(client_list[event['group']].index(event['client']))
        await self.send(text_data=json.dumps({
            'message': message
        }))

        '''if len(client_list[self.room_name]) == 2 and not play_game:
            with open('action.txt','w'): #reset
                pass'''
        if (len(temp) == 3):
            if(temp[0] == 'play' and temp[1] == 'game'and not play_game):
                with open('action.txt','w'): #reset
                        pass
                if(temp[2] == 'random'):
                    asyncio.create_task(run.play(-1,'human','rand',socket=self))
                elif(temp[2] == 'greedy'):
                    asyncio.create_task(run.play(-1,'human','greedy',socket=self))
                elif(temp[2] == 'maximin'):
                    asyncio.create_task(run.play(-1,'human','maximin',socket=self))
                elif(temp[2] == 'DQN'):
                    asyncio.create_task(run.play(-1,'human','DQN',socket=self))
                play_game = True

    async def send_gameboard(self, board):
        global count
        if len(str(board))<11:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': str(board),
                    'group' : self.room_name
                }
            )
        elif (board[:11] =="Turn: BLACK" and count > 1 and play_game):
            count = 0
        else:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': board,
                    'group' : self.room_name
                }
            )
            count += 1


        

