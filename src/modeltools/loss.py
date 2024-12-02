import tensorflow as tf


EARTH_RAD = 6371.0
def haversine_loss_batch(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    lat_true, lon_true = tf.split(tf.math.radians(y_true), 2, axis=1)
    lat_pred, lon_pred = tf.split(tf.math.radians(y_pred), 2, axis=1)
    delta_lat, delta_lon = lat_pred - lat_true, lon_pred - lon_true
    a = tf.sin(delta_lat / 2) ** 2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(delta_lon / 2) ** 2
    return tf.reduce_mean(EARTH_RAD * 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a)))

def haversine_loss_single(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    lat_true, lon_true = tf.split(tf.math.radians(y_true), 2, axis=1)
    lat_pred, lon_pred = tf.split(tf.math.radians(y_pred), 2, axis=1)
    delta_lat, delta_lon = lat_pred - lat_true, lon_pred - lon_true
    a = tf.sin(delta_lat / 2) ** 2 + tf.cos(lat_true) * tf.cos(lat_pred) * tf.sin(delta_lon / 2) ** 2
    return EARTH_RAD * 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))