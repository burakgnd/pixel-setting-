import struct
from pymavlink import mavutil

# USB modem bağlantısını başlat
mav = mavutil.mavlink_connection('/dev/tty.usbmodem1401', baud=57600)

def decode_gps_raw_int(message):
    # GPS_RAW_INT mesajını çöz
    time_usec = message.time_usec
    lat = message.lat
    lon = message.lon
    alt = message.alt
    eph = message.eph
    epv = message.epv
    vel = message.vel
    cog = message.cog
    fix_type = message.fix_type
    satellites_visible = message.satellites_visible
    
    return {
        'time_usec': time_usec,  # Zaman (mikrosaniye)
        'lat': lat / 1e7,        # Enlem (derece)
        'lon': lon / 1e7,        # Boylam (derece)
        'alt': alt / 1e3,        # Yükseklik (metre)
        'eph': eph,              # GPS HDOP
        'epv': epv,              # GPS VDOP
        'vel': vel,              # GPS yer hızı (cm/s)
        'cog': cog / 100,        # Yerden varış açısı (derece)
        'fix_type': fix_type,    # GPS fix tipi
        'satellites_visible': satellites_visible  # Görünür uydu sayısı
    }

# Ana döngü
while True:
    # Mesaj almak için blokla
    msg = mav.recv_match(blocking=True)
    if msg is not None and msg.get_type() == 'GPS_RAW_INT':
        # Mesajın yükünü çöz
        decoded_payload = decode_gps_raw_int(msg)
        if decoded_payload:
            print('Decoded Payload:', decoded_payload)
