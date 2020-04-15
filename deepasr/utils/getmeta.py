import sys
from tinytag import TinyTag


# https://pypi.org/project/tinytag/

def get_file_tags(audio_file):
    tag = TinyTag.get(audio_file)
    print(tag.filesize, '|', tag.audio_offest, "|", tag.bitrate, "|", tag.channels, "|", tag.duration, "|",
          tag.samplerate, "|", tag.audio_offset)


if __name__ == "__main__":
    get_file_tags(sys.argv[1])
