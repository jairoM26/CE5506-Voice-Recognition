from pydub import AudioSegment

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

sound = AudioSegment.from_file("datos_proyecto2/catorce_2_2.wav", "wav")
normalized_sound = match_target_amplitude(sound, -20.0)
normalized_sound.export("nomrmalizedAudio3.wav", format="wav")
