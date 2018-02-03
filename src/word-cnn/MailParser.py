import email.header

class MailParser:

    @staticmethod
    def getMinimalMessageHeader(email):
        header_str = ""
        for header in email.keys():
            value = email[header]
            if header == "From":
                header_str = header_str + "MYFROM " + MailParser.decodeUtf(email['from']) + "\n"
            if header == "To":
                header_str = header_str + "MYTO " + MailParser.decodeUtf(email['to']) + "\n"
            if header == "Cc":
                header_str = header_str + "MYCC " + MailParser.decodeUtf(email['cc']) + "\n"
            if header == "Subject":
                header_str = header_str + "MYSUBJECT " + MailParser.decodeUtf(email['subject']) + "\n"
            if header == "References":
                header_str = header_str + "MYREF " + MailParser.decodeUtf(email['references']) if email['references'] else 'NONE'
                header_str = header_str + "\n"
        return header_str

    @staticmethod
    def getMessageHeader(email):
        header_str = MailParser.getMinimalMessageHeader(email)
        for header in email.keys():
            value = email[header]
            header_str = header_str + MailParser.decodeUtf(header) + " " + MailParser.decodeUtf(value) + "\n"
        return header_str

    @staticmethod
    def parseMessage(message):
        ctype = message.get_content_type()
        cdispo = str(message.get('Content-Disposition'))
        charset = message.get_charset()
        c_charset = message.get_content_charset()
        body = ""
        # parse text/plain (txt) parts that do not belong to attachments
        if ctype == 'text/plain' and 'attachment' not in cdispo:
            if c_charset:
                body1 = message.get_payload(decode=True)
                body1 = body1.decode(c_charset)
            else:
                body1 = message.get_payload()

            body = body + body1 + "\n"
        else:
            # report ctype only
            body = body + " " + ctype + "\n"
        return body

    @staticmethod
    def parseMessageParts(message):
        if message.is_multipart():
            body = ""
            for part in message.walk():
                body = body + MailParser.parseMessage(part)
        else:
            body = MailParser.parseMessage(message)
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

