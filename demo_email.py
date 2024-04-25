from email.header import Header
from email.mime.text import MIMEText
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication 
from email.utils import parseaddr, formataddr



receivers = ['chenjiamu@ksquant.com.cn','657288788@qq.com','ftluo@ksquant.com.cn']
#receivers = ['chenjiamu@ksquant.com.cn']
#在调用时import receivers，然后使用append，remove修改     
#receivers.remove('657288788@qq.com')
#receivers.remove('ftluo@ksquant.com.cn')

def _format_addr(s):
    addr = parseaddr(s)
    return formataddr(addr)

def send_mail(msg, images = None, files = None, file_names = None, content_msg = None):
    sender = 'chenjiamu@ksquant.com.cn' # 你的邮箱
    message = MIMEMultipart('related')
    message['From'] = _format_addr(u'jmchen <%s>' % sender) 
    message['To'] = ','.join(receivers)
    header_msg = msg
    if len(msg) > 80:
        header_msg = msg[0:80] + ' ......'
    message['Subject'] = Header(header_msg, 'utf-8')
    if images:
        pic_inline = '<p> </p>'
        for index,pic_file in enumerate(images):
            pic_file_name = msg
            with open(pic_file,'rb') as image:
                image_info = MIMEImage(image.read())
                image_info.add_header('Content-Id',f'<image{index+1}>')
                message.attach(image_info)
                tmp_pic_inline = f'''
                <p>{msg}</p>
                    <!-- <br>  {pic_file_name}:</br> -->
                    <br><img src="cid:image{index+1}" width="300" alt={pic_file_name}></br>
                    '''
                pic_inline += tmp_pic_inline+'\n'
        message.attach(MIMEText(pic_inline, "html", "utf-8"))
    else:
        message_text = MIMEText(content_msg, 'plain', 'utf-8')
        message.attach(message_text)
    if files:
        for file_path, file_name in zip(files, file_names):
            file = MIMEApplication(open(file_path, 'rb').read())
            file.add_header('Content-Disposition', 'attachment', filename = '%s'%file_name)
            message.attach(file)


    try:
        smtpObj = smtplib.SMTP_SSL("157.148.45.129")
        smtpObj.connect("157.148.45.129", 465)

        smtpObj.login(sender, "MHy3xrAb9TAYZsJB")   #企业邮箱的授权码
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功" + msg)
    except smtplib.SMTPException:
        print("邮件未能发送")

def send_mail_index(msg, images = None, names = None, files = None, file_names = None, msg_content = None): # index_data_monitors.py
    global receivers
    sender = 'chenjiamu@ksquant.com.cn' 
    message = MIMEMultipart('related')
    message['From'] = _format_addr(u'jmchen <%s>' % sender)                
    print(','.join(receivers))
    message['To'] = ','.join(receivers)
    header_msg = msg
    if len(msg) > 80:
        header_msg = msg[0:80] + ' ......'
    message['Subject'] = Header(header_msg, 'utf-8')
    name_index = 0
    if images:
        pic_inline = '<p> </p>'
        if msg_content:
            pic_inline = msg_content + '<p> </p>'
        for index,pic_file in enumerate(images):
            pic_file_name = names[name_index]
            name_index = name_index + 1
            with open(pic_file,'rb') as image:
                image_info = MIMEImage(image.read())
                image_info.add_header('Content-Id',f'<image{index+1}>')
                message.attach(image_info)
                tmp_pic_inline = f'''
                <p>{pic_file_name}</p>
                    <!-- <p>  {pic_file_name}:</p> -->
                    <br><img src="cid:image{index+1}" width="300" alt={pic_file_name}></br>
                    '''
                pic_inline += tmp_pic_inline
        message.attach(MIMEText(pic_inline, "html", "utf-8"))
    else:
        if msg_content:
            message.attach(MIMEText(msg_content, "html", "utf-8"))
        # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
        else:
            message_text = MIMEText(msg, 'plain', 'utf-8')
            message.attach(message_text)

    if files:
        for file_path, file_name in zip(files, file_names):
            file = MIMEApplication(open(file_path, 'rb').read())
            file.add_header('Content-Disposition', 'attachment', filename = '%s'%file_name)
            message.attach(file)
    try:
        print("ready to connect")
        # smtpObj = smtplib.SMTP()
        smtpObj = smtplib.SMTP_SSL("157.148.45.129")  #     157.148.45.129
        print('SMTP_SSL')
        smtpObj.connect("157.148.45.129", 465)  # smtp.qq.com
        print("success_connect")
        # WAIMFZYOIEZHOZIQ
        smtpObj.login(sender, "MHy3xrAb9TAYZsJB")   #企业邮箱的授权码
        smtpObj.sendmail(sender, receivers, message.as_string())
        smtpObj.quit()
        print("邮件发送成功" + msg)
    except smtplib.SMTPException:
        print("邮件未能发送")

#send_mail( msg='test3',files =['/home/jmchen/files/demo(1)(2).py'], file_names =['test_file'],content_msg='邮箱测试！\n 换行')


#recievers=['chenjiamu@ksquant.com.cn','cyt@ksquant.com.cn']
#send_mail_index('net_value信息', images =['/home/jmchen/test/test_net/index_weight/compound_interest.png'] , names = ['compound_interest.png'], files=['/home/jmchen/test/test_net/index_weight/index_info.csv'],file_names=[''])

# receivers.remove('657288788@qq.com')
# receivers.remove('ftluo@ksquant.com.cn')
# send_mail_index('net_value信息')