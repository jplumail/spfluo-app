import locale
import sys

NO_INDEX = 0  # This is the index value for single images

if sys.version_info >= (3, 11):

    def getencoding():
        return locale.getencoding()

else:

    def getencoding():
        return locale.getpreferredencoding(False)

MICRON_STR_UTF_8 = "µm"

if getencoding() == "UTF-8":
    MICRON_STR = MICRON_STR_UTF_8
else:
    MICRON_STR = "um"
