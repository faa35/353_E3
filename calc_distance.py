from pykalman import KalmanFilter


import numpy as np
import pandas as pd
import sys


from xml.etree import ElementTree
# source: https://docs.python.org/3/library/xml.etree.elementtree.html


def get_data(filename):
    # print("get_data is reading:", input_gpx)

    tree = ElementTree.parse(filename)
    namespaces = '{http://www.topografix.com/GPX/1/0}'
    rows = []

   # You will need to extract the latitude and longitude and time from each <trkpt> element. We can ignore the elevation and other fields. Create a DataFrame with columns 'datetime', 'lat' and 'lon' holding the observations. [It's certainly possible to do this without loops, but you may write a loop to iterate as you read the file/elements for this part.]
   # doing this with loop:
    for trkpt in tree.iter(f'{namespaces}trkpt'): 
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))

        time_text = trkpt.find(f'{namespaces}time').text

        rows.append({'datetime': time_text, 'lat': lat, 'lon': lon})


    data = pd.DataFrame(rows)

    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

    return data


def distance(points):
    #the haversine formul
    # source: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206

    #so for R*C we meed earth radius in meters

    R = 6371000 # source: https://www.mathworks.com/help/map/ref/earthradius.html

    # current point
    lat1 = np.radians(points['lat'])
    lon1 = np.radians(points['lon'])

    #previous point
    lat2 = np.radians(points['lat'].shift())
    lon2 = np.radians(points['lon'].shift())

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).sum()


def smooth(points):
    #Kalman predictions rely on a “prediction”. Our prediction for the “next” values will be:
    transition = [
        [1, 0,  5e-7,  34e-7],
        [0, 1, -49e-7,  9e-7],
        [0, 0,  1,      0   ],
        [0, 0,  0,      1   ],
    ]

    #Around Vancouver, one degree of latitude or longitude is about  meters. That will be a close enough conversion as we're estimating error…
    # While GPS can be much more accurate, we assume it is accurate to about 5 metres. Artificial noise has been added to the data to make this problem possible. (This implies a value for observation_covariance.)
    obs_std_latlon = 5.0 / 1e5   # =5e-5 degre
    obs_std_B = 5.0      #could be 5 noise in the Bx/By measurements


    observation_covariance = np.diag([
        obs_std_latlon, obs_std_latlon, obs_std_B, obs_std_B
    ]) ** 2

    # lat/lon: pace changes cause prediction error, but model is good= small
    # Bx/By:  "they don't change" is a poor prediction= larger noise
    transition_covariance = np.diag([
        obs_std_latlon * 0.5,   
        obs_std_latlon * 0.5,   
        obs_std_B * 2,         
        obs_std_B * 2,          
    ]) ** 2

    initial_state = points.iloc[0]

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition,
    )
    smoothed, _ = kf.smooth(points)
    return pd.DataFrame(smoothed, columns=points.columns, index=points.index)


def output_gpx(points, output_filename):
    from xml.dom.minidom import getDOMImplementation

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % pt['lat'])
        trkpt.setAttribute('lon', '%.7f' % pt['lon'])
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def main():
    input_gpx = sys.argv[1]
    input_csv = sys.argv[2]

    points = get_data(input_gpx).set_index('datetime')
    sensor_data = pd.read_csv(input_csv, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']

    dist = distance(points)
    print(f'Unfiltered distance: {dist:.2f}')

    smoothed_points = smooth(points)
    smoothed_dist = distance(smoothed_points)
    print(f'Filtered distance: {smoothed_dist:.2f}')

    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
