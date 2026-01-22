import styles from './Find.module.css';
import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";

const containerStyle = {
  width: "100%",
  height: "400px",
};

const center = {
  lat: 40.7128,
  lng: -74.0060,
};

export default function Find() {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
  console.log("Google Maps API Key:", apiKey); // Debugging line
  return (
    <div className={styles.findContainer}>
      <h1>Find</h1>
      <p>This is the Find page.</p>

      <div className={styles.mapContainer}>
        <LoadScript googleMapsApiKey={apiKey}>
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={center}
            zoom={10}
          >
            <Marker position={center} />
          </GoogleMap>
        </LoadScript>
      </div>
    </div>
    
  )
}
