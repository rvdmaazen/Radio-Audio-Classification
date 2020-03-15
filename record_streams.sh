#!/bin/bash
RECORD_LEN=600
TODAY=`date +%Y%m%d.%H%M`

# Qmusic
ffmpeg -i https://icecast-qmusicnl-cdp.triple-it.nl/Qmusic_nl_live_32.aac -t $RECORD_LEN audio/original/q-music_$TODAY.wav &

# Qmusic Non-Stop
ffmpeg -i https://icecast-qmusicnl-cdp.triple-it.nl/Qmusic_nl_nonstop_32.aac -t $RECORD_LEN audio/original/q-music-non-stop_$TODAY.wav &

# Sky Radio
ffmpeg -i https://21293.live.streamtheworld.com/SKYRADIO.mp3 -t $RECORD_LEN audio/original/sky-radio_$TODAY.wav &

# Sky Radio Non-Stop
ffmpeg -i https://21283.live.streamtheworld.com/SRGSTR24.mp3 -t $RECORD_LEN audio/original/sky-radio-non-stop_$TODAY.wav &

# Radio 538
ffmpeg -i https://19093.live.streamtheworld.com/RADIO538AAC.aac -t $RECORD_LEN audio/original/radio-538_$TODAY.wav &

# Radio 538 Non-Stop
ffmpeg -i https://21323.live.streamtheworld.com/TLPSTR09AAC.aac -t $RECORD_LEN audio/original/radio-538-non-stop_$TODAY.wav &

# NPO Radio 1
ffmpeg -i https://icecast.omroep.nl/radio1-bb-mp3 -t $RECORD_LEN audio/original/radio-1_$TODAY.wav &

# NPO Radio 2
ffmpeg -i https://icecast.omroep.nl/radio2-bb-mp3 -t $RECORD_LEN audio/original/radio-2_$TODAY.wav &

# NPO Radio 3FM
ffmpeg -i https://icecast.omroep.nl/3fm-bb-mp3 -t $RECORD_LEN audio/original/radio-3fm_$TODAY.wav &

# SLAM!
ffmpeg -i https://21273.live.streamtheworld.com/SLAM_AAC.aac -t $RECORD_LEN audio/original/slam-fm_$TODAY.wav &

# Radio Veronica
ffmpeg -i https://21263.live.streamtheworld.com/VERONICAAAC.aac -t $RECORD_LEN audio/original/veronica_$TODAY.wav &