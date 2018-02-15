import codecs
import gnupg
gpg = gnupg.GPG(gnupghome='/home/robert/.gnupg')


class Decryptor:

    def __init__(self, filename='pphrase.txt'):
        self.__pphrase = self.load_passphrase(filename)


    def decrypt(self, data):
        decrypted_data = gpg.decrypt(data, passphrase=self.__pphrase)
        return decrypted_data.data


    def load_passphrase(self, filename):
        with open(filename, 'r') as file:
            text = file.read().strip()
            passphrase = codecs.encode(text, 'rot_13')
        return passphrase

