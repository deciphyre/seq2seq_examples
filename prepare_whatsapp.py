import sys

whatsapp_file = sys.argv[1]

with open(whatsapp_file) as w:
    ignore = ['<Media omitted>']
    message = []
    prev_message = None
    for line in w:
        line = line.strip()
        if len(line) > 0:
            line = line.split(':')
            #print(line, len(line))
            msg = line[-1].strip()

            if len(line) > 1:
                if len(message) > 0 :
                    message = " <eos> ".join(message)
                    if prev_message:
                        print(prev_message + "\t"+ message)
                    prev_message = message
                message = []
            if msg not in ignore:
                message.append(msg)

            #message = line[1].strip()
            #if message not in ignore:
            #    print(message)