import email.header
from email import message_from_bytes
import string
import decryptGnupg


class MailParser:


    def __init__(self, use_gpg=False):
        self.decryptor = None
        if use_gpg:
            self.decryptor = decryptGnupg.Decryptor()

    @staticmethod
    def unifyReferences(ref_str):
        REF_TOKEN = 'EMAILREFID'
        ref_cnt = len(ref_str.split())
        out_string = ' '.join([REF_TOKEN + string.ascii_uppercase[i] for i in range(min(26, ref_cnt))])
        return out_string

    @staticmethod
    def getMinimalMessageHeader(email):
        header_str = ""
        for header in email.keys():
            value = email[header]
            if header == 'From':
                header_str = header_str + "MYFROM " + MailParser.decodeUtf(email['from']) + "\n"
            if header == 'To':
                header_str = header_str + "MYTO " + MailParser.decodeUtf(email['to']) + "\n"
            if header == 'Cc':
                header_str = header_str + "MYCC " + MailParser.decodeUtf(email['cc']) + "\n"
            if header == 'Date':
                header_str = header_str + "MYDATE" + MailParser.decodeUtf(email['Date']) + "\n"
            if header == 'Subject':
                header_str = header_str + "MYSUBJECT " + MailParser.decodeUtf(email['subject']) + "\n"
            if header == 'References':
                if email['references']:
                    ref_str = MailParser.unifyReferences(MailParser.decodeUtf(email['references']))
                    header_str = header_str + 'MYREF ' + ref_str + "\n"
        return header_str

    @staticmethod
    def getMessageHeader(email):
        header_str = MailParser.getMinimalMessageHeader(email)
        for header in email.keys():
            value = email[header]
            header_str = header_str + MailParser.decodeUtf(header) + " " + MailParser.decodeUtf(value) + "\n"
        return header_str

    def parseMessage(self, message, start):
        ctype = message.get_content_type()
        cdispo = str(message.get('Content-Disposition'))
        charset = message.get_charset()
        c_charset = message.get_content_charset()
        body = ""
        # parse text/plain (txt) parts that do not belong to attachments
        if ctype == 'text/plain' and 'attachment' not in cdispo:
            if c_charset:
                body1 = message.get_payload(decode=True)
                try:
                    body1 = body1.decode(c_charset)
                except UnicodeDecodeError:
                    body1 = 'ERROR-' + c_charset
            else:
                body1 = message.get_payload()
            body = body + body1 + "\n"
        elif ctype == 'multipart/mixed':
            # nested multipart-messages are tricky
            # message.walk does not walk into nested multipart/mixed messages
            # so we do it manually here...
            if not start:
                payload = message.get_payload()
                if isinstance(payload, list):
                    body1 = ""
                    for msg in payload:
                        body1 += "payload" #self.parseMessage(msg, start=False)
                else:
                    body1 = ctype
            else:
                body1 = ctype
            body = body + body1 + "\n"
        elif ctype == 'application/pgp-encrypted' or ctype == 'application/octet-stream':
            body1 = message.get_payload()
            if body1.split("\n")[0] == '-----BEGIN PGP MESSAGE-----':
                crypt_data = self.decryptor.decrypt(body1)
                msg = message_from_bytes(crypt_data)
                body1 = self.parseMessageParts(msg)
            body = body + body1 + "\n"
        else:
            # report ctype only
            body = body + " " + ctype + "\n"
        return body

    def parseMessageParts(self, message):
        if message.is_multipart():
            body = ""
            start = True
            for part in message.walk():
                body = body + self.parseMessage(part, start)
                start = False
        else:
            body = self.parseMessage(message, True)
        return body


    @staticmethod
    def decodeUtf(text):
        x = text
        try:
            for word, encoding in email.header.decode_header(text):
                if isinstance(word, bytes) and encoding not in ['unknown-8bit']:
                    x = u''.join(word.decode(encoding or 'utf8'))
                else:
                    x = str(word)
        except TypeError:
            pass
        return x

