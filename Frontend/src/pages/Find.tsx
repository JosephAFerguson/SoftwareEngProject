import { useRef, useState } from "react";
import styles from './Find.module.css';
import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";

type Listing = {
  id: number;
  title: string;
  price: number;
  address: string;
  lat: number;
  lng: number;
};

const listings: Listing[] = [
  {
    id: 1,
    title: "Private Room Near UC",
    price: 700,
    address: "235 Calhoun St, Cincinnati, OH",
    lat: 39.1302,
    lng: -84.5150,
  },
  {
    id: 2,
    title: "Studio on Short Vine",
    price: 950,
    address: "2724 Short Vine St, Cincinnati, OH",
    lat: 39.1286,
    lng: -84.5144,
  },
  {
    id: 3,
    title: "Shared Apartment",
    price: 600,
    address: "305 Warner St, Cincinnati, OH",
    lat: 39.1273,
    lng: -84.5186,
  },
  {
    id: 4,
    title: "2BR Apartment",
    price: 1200,
    address: "2900 Jefferson Ave, Cincinnati, OH",
    lat: 39.1316,
    lng: -84.5198,
  },
  {
    id: 5,
    title: "Basement Unit",
    price: 550,
    address: "2601 Highland Ave, Cincinnati, OH",
    lat: 39.1279,
    lng: -84.5209,
  },
  {
    id: 6,
    title: "Modern Condo",
    price: 1400,
    address: "2510 Clifton Ave, Cincinnati, OH",
    lat: 39.1329,
    lng: -84.5216,
  },
];

const containerStyle = {
  width: "100%",
  height: "100%",
};

const defaultCenter = {
  lat: 39.1310,
  lng: -84.5165,
};

export default function Find() {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  const [selectedListing, setSelectedListing] = useState<Listing | null>(null);
  const mapRef = useRef<google.maps.Map | null>(null);

  const handleSelectListing = (listing: Listing) => {
    setSelectedListing(listing);

    if (mapRef.current) {
      mapRef.current.panTo({ lat: listing.lat, lng: listing.lng });
      mapRef.current.setZoom(15);
    }
  };

  return (
    <div className={styles.findContainer}>

      {/* LISTINGS */}
      <div className={styles.listingsContainer}>
        <h2>Listings Near UC</h2>

        <div className={styles.listingsGrid}>
          {listings.map((listing) => (
            <div
              key={listing.id}
              className={`${styles.listingCard} ${
                selectedListing?.id === listing.id ? styles.active : ""
              }`}
              onClick={() => handleSelectListing(listing)}
            >
              <h3>{listing.title}</h3>
              <p>{listing.address}</p>
              <p className={styles.price}>${listing.price} / month</p>
            </div>
          ))}
        </div>
      </div>

      {/* MAP */}
      <div className={styles.mapContainer}>
        <LoadScript googleMapsApiKey={apiKey}>
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={selectedListing ?? defaultCenter}
            zoom={14}
            onLoad={(map) => {
              mapRef.current = map;
            }}
          >
            {listings.map((listing) => (
              <Marker
                key={listing.id}
                position={{ lat: listing.lat, lng: listing.lng }}
                title={listing.title}
                onClick={() => handleSelectListing(listing)}
              />
            ))}
          </GoogleMap>
        </LoadScript>
      </div>

    </div>
  );
}