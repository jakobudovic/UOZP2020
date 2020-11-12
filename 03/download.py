from os import listdir

import wikipedia

link = "https://el.wikipedia.org/wiki/Ιωάννης_Καποδίστριας"

# for outputting the right format


names = {
    "de": "german",
    "sl": "slovenian",
    "it": "italian",
    "es": "spanish",
    "fr": "french",
    "ja": "japanese",
    "ru": "russian",
    "zh": "chinese",
    "uk": "ukrainian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "sv": "swedish",
    "ro": "romanian",
    "da": "danish",
    "sr": "serbian",
    "hr": "croatian",
    "el": "greek",
    "bs": "bosnian",
    "no": "norwegian",
}

count = {}

# check existing files to increase counter

for fn in listdir("data"):
    name = fn.split("_")[0]
    if name not in count:
        count[name] = 1
    else:
        count[name] += 1

# handle the link

prefix = link[8:10]

split_link = link.split("/")

wikipedia.set_lang(prefix)
p = wikipedia.page(split_link[len(split_link)-1])

length = 14000

if prefix == "ja" or prefix == "zh":
    length = 7000

string = p.content[0:length]

filename = names[prefix] + "_" + str(count[names[prefix]]+1)

text = open("C:/Users/rok/Desktop/faks/uozp/naloga3/data/" + filename + ".txt", "w", encoding="utf-8")
text.write(string)
text.close()