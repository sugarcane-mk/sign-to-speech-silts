# #!/bin/bash

# # Check if argument is provided
# if [ -z "$1" ]; then
#   echo "Usage: ./run_all.sh [webcam|video_file]"
#   exit 1
# fi

# echo "Running gesture recognition with input: $1"

# # Run inference and save stdout and stderr to output.txt
# python3 inference.py "$1" 1>&2 | tee output.txt

# echo -e "\n\n"

# # Read the prediction text from output.txt (trim whitespace/newlines)
# text=$(tr -d '\r\n' < output.txt)

# echo "Converting \"$text\" to speech...."

# source activate_python.sh
# # Pass the text as an argument to the synthesis script (quoted)
# python3 synthesize_CONTROL.py "$text"

# # Play the generated audio file
# aplay test.wav

rm test.wav
rm output.txt

# Check for input argument
if [ -z "$1" ]; then
  echo "Usage: $0 [webcam|video_file]"
  exit 1
fi

echo "Running gesture recognition with input: $1"

# Run inference.py and save output (stdout and stderr) to output.txt
python3 inference.py $1  > output.txt

echo -e "\n\n"

# Read prediction text from output.txt and trim newlines
text=$(cat output.txt)

echo "Converting $text to speech...."


source activate_python.sh
# Call synthesis script with the prediction text
#python3 synthesize_CONTROL.py $text

python3 g_tts.py $text



# Play the generated audio (Linux only)
# aplay test.wav
