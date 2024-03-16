from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "_______________" #git 업로드해야되서 보안상 지웠음
# Your Auth Token from twilio.com/console
auth_token  = "________________" #git 업로드해야되서 보안상 지웠음

client = Client(account_sid, auth_token)

message = client.messages.create(
    to="_____________", #전화번호넣기
    from_="____________",   #거는사람전화번호넣기
    body="숭실대 정보과학관 601호에 마스크 미착용자가 있어요!")

print(message.sid)