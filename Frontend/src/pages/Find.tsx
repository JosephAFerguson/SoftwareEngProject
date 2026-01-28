// ============================================================================
// IMPORTS
// ============================================================================

import { useRef, useState } from "react";
import styles from './Find.module.css';
import { GoogleMap, Marker, useJsApiLoader } from "@react-google-maps/api";
import { FiMinimize2 , FiSearch} from "react-icons/fi";

import pic1 from "../../public/pic1.png";
import pic2 from "../../public/pic2.png";
import pic3 from "../../public/pic3.png";
import pic4 from "../../public/pic4.png";
import pic5 from "../../public/pic5.png";

// ============================================================================
// TYPES
// ============================================================================

/**
 * Represents a rental listing with all relevant information
 */
type Listing = {
  id: number;
  title: string;
  price: number;
  address: string;
  lat: number;
  lng: number;
  imageUrl: string;
};

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Mock data for rental listings
 * TODO: Replace with API call to Backend service
 */
const listings: Listing[] = [
  {
    id: 1,
    title: "Private Room Near UC",
    price: 700,
    address: "235 Calhoun St, Cincinnati, OH",
    lat: 39.1302,
    lng: -84.5150,
    imageUrl: pic1,
  },
  {
    id: 2,
    title: "Studio on Short Vine",
    price: 950,
    address: "2724 Short Vine St, Cincinnati, OH",
    lat: 39.1286,
    lng: -84.5144,
    imageUrl: pic2,
  },
  {
    id: 3,
    title: "Shared Apartment",
    price: 600,
    address: "305 Warner St, Cincinnati, OH",
    lat: 39.1273,
    lng: -84.5186,
    imageUrl: pic3,
  },
  {
    id: 4,
    title: "2BR Apartment",
    price: 1200,
    address: "2900 Jefferson Ave, Cincinnati, OH",
    lat: 39.1316,
    lng: -84.5198,
    imageUrl: pic4,
  },
  {
    id: 5,
    title: "Basement Unit",
    price: 550,
    address: "2601 Highland Ave, Cincinnati, OH",
    lat: 39.1279,
    lng: -84.5209,
    imageUrl: pic5,
  },
  {
    id: 6,
    title: "Modern Condo",
    price: 1400,
    address: "2510 Clifton Ave, Cincinnati, OH",
    lat: 39.1329,
    lng: -84.5216,
    imageUrl: pic1,
  },
];

/**
 * Google Map container styling
 */
const containerStyle = {
  width: "100%",
  height: "100%",
};

/**
 * Default map center coordinates (Cincinnati, OH)
 */
const defaultCenter = {
  lat: 39.1310,
  lng: -84.5165,
};

// ============================================================================
// COMPONENT
// ============================================================================

/**
 * Find Component
 * Displays a list of rental listings and an interactive map
 * Features:
 * - Click to expand individual listing cards
 * - Hover to preview listings on the map
 * - Minimize icon to collapse expanded cards
 */
export default function Find() {
  // Get API key from environment variables
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  // ========================================================================
  // STATE
  // ========================================================================

  const [selectedListing, setSelectedListing] = useState<Listing | null>(null);
  const [lockedListingId, setLockedListingId] = useState<number | null>(null);
  const mapRef = useRef<google.maps.Map | null>(null);
  const [zoom, setZoom] = useState(14);
  const [filterExpanded, setFilterExpanded] = useState(false);
  const [minPrice, setMinPrice] = useState<number>(0);
  const [maxPrice, setMaxPrice] = useState<number>(2000);
  const [searchQuery, setSearchQuery] = useState<string>("");

  // ========================================================================
  // EVENT HANDLERS
  // ========================================================================

  /**
   * Updates the selected listing and centers the map on it
   * @param listing - The listing to select
   */
  const handleSelectListing = (listing: Listing) => {
    setSelectedListing(listing);
    setZoom(15);

    if (mapRef.current) {
      mapRef.current.panTo({ lat: listing.lat, lng: listing.lng });
    }
  };

  /**
   * Handles hover over a listing card
   * Only updates selection if no card is currently locked/expanded
   * @param listing - The listing being hovered over
   */
  const handleHoverListing = (listing: Listing) => {
    if (lockedListingId !== null) return;
    handleSelectListing(listing);
  };

  /**
   * Handles click on a listing card
   * Expands the card and locks it to prevent hover deselection
   * @param listing - The listing being clicked
   */
  const handleClickListing = (listing: Listing) => {
    if (lockedListingId === listing.id) {
      // Don't unlock on card click - only via minimize icon
      return;
    } else {
      // Lock the card and expand it
      setLockedListingId(listing.id);
      setFilterExpanded(false); // Hide filter when expanding card
      handleSelectListing(listing);

      // Scroll the expanded card into view
      const element = document.getElementById(`listing-${listing.id}`);
      element?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  /**
   * Handles minimize icon click
   * Closes the expanded card and resets map view
   * @param e - React mouse event
   */
  const handleMinimize = (e: React.MouseEvent) => {
    e.stopPropagation();
    setLockedListingId(null);
    setSelectedListing(null);
    setZoom(14);
  };

  /**
   * Toggles the filter panel open/closed
   */
  const handleToggleFilter = () => {
    setFilterExpanded(!filterExpanded);
  };

  /**
   * Applies filters and closes the filter panel
   */
  const handleApplyFilters = () => {
    setFilterExpanded(false);
    // TODO: Filter the listings array based on the filter criteria
  };

  /**
   * Resets all filters to default values
   */
  const handleResetFilters = () => {
    setMinPrice(0);
    setMaxPrice(2000);
    setSearchQuery("");
  };

  // ========================================================================
  // API CALLS
  // ========================================================================

  /**
   * Fetches listing data from the backend API
   * TODO: Implement and integrate with real data
   */
  const grabListingData = async () => {
    const res = await fetch("http://localhost:3000/api/v1/listings");
    const data = await res.json();
    console.log(data);
  };

  // ========================================================================
  // GOOGLE MAPS
  // ========================================================================

  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: apiKey,
  });

  // ========================================================================
  // RENDER
  // ========================================================================

  return (
    <div className={styles.findContainer}>
      {/* ============================================================== */}
      {/* LISTINGS SECTION (Left Column) */}
      {/* ============================================================== */}
      <div className={styles.listingsContainer}>
        {/* Filter panel (hidden when card is expanded) */}
        {/* ============================================================== */}
        {lockedListingId === null && (
          <div className={styles.filterSlot}>
            <div
              className={`${styles.listingsFilter} ${
                filterExpanded ? styles.filterExpanded : ""
              }`}
              onClick={!filterExpanded ? handleToggleFilter : undefined}
            >
              {/* Filter header */}
              <div className={styles.filterHeader}>
                <div className={styles.filterHeaderLeft}>
                  <FiSearch size={20} />
                  <span>Filters</span>
                </div>

                {filterExpanded && (
                  <FiMinimize2
                    size={20}
                    className={styles.filterMinimizeIcon}
                    onClick={(e) => {
                      e.stopPropagation();
                      setFilterExpanded(false);
                    }}
                    title="Close filters"
                  />
                )}
              </div>

              {/* Expanded filter options */}
              {filterExpanded && (
                <div className={styles.filterOptions}>
                  {/* Search */}
                  <div className={styles.filterGroup}>
                    <label htmlFor="search-input">Search</label>
                    <input
                      id="search-input"
                      type="text"
                      placeholder="Search listings..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className={styles.filterInput}
                    />
                  </div>

                  {/* Price range */}
                  <div className={styles.filterGroup}>
                    <label>Price Range</label>
                    <div className={styles.priceRangeContainer}>
                      <div className={styles.priceInput}>
                        <label htmlFor="min-price">Min</label>
                        <input
                          id="min-price"
                          type="number"
                          min="0"
                          max={maxPrice}
                          value={minPrice}
                          onChange={(e) => setMinPrice(Number(e.target.value))}
                          className={styles.filterInput}
                        />
                      </div>

                      <span className={styles.priceSeparator}>-</span>

                      <div className={styles.priceInput}>
                        <label htmlFor="max-price">Max</label>
                        <input
                          id="max-price"
                          type="number"
                          min={minPrice}
                          value={maxPrice}
                          onChange={(e) => setMaxPrice(Number(e.target.value))}
                          className={styles.filterInput}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className={styles.filterActions}>
                    <button
                      className={styles.resetButton}
                      onClick={handleResetFilters}
                    >
                      Reset
                    </button>
                    <button
                      className={styles.applyButton}
                      onClick={handleApplyFilters}
                    >
                      Apply Filters
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}


        {/* Listings grid (hidden when filter is expanded) */}
          <div
            className={`${styles.listingsGrid} ${
              lockedListingId !== null ? styles.hasExpanded : ""
            }`}
          >
          {/* Map through listings and render cards */}
          {listings.map((listing) => (
            <div
              key={listing.id}
              id={`listing-${listing.id}`}
              className={`${styles.listingCard}
                ${selectedListing?.id === listing.id ? styles.active : ""}
                ${lockedListingId === listing.id ? styles.expanded : ""}
              `}
              onMouseEnter={() => handleHoverListing(listing)}
              onClick={() => handleClickListing(listing)}
            >
              <div className={styles.cardContent}>
                {/* Card header with title and minimize icon */}
                <div className={styles.cardContentHeader}>
                  <h3>{listing.title}</h3>
                  {lockedListingId === listing.id && (
                    <FiMinimize2
                      size={24}
                      className={styles.minimizeIcon}
                      onClick={(e) => handleMinimize(e)}
                      title="Minimize"
                    />
                  )}
                </div>

                {/* Listing image */}
                <img
                  src={listing.imageUrl}
                  alt={listing.title}
                  className={styles.listingImage}
                />

                {/* Listing details */}
                <p>{listing.address}</p>
                <p className={styles.price}>${listing.price} / month</p>

                {/* Expanded details (only shown when card is maximized) */}
                {lockedListingId === listing.id && (
                  <div className={styles.expandedDetails}>
                    <h4>Listing Details</h4>
                    <p>More information about the listing can go here.</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ============================================================== */}
      {/* MAP SECTION (Right Column) */}
      {/* ============================================================== */}
      <div className={styles.mapContainer}>
        {isLoaded && (
          <GoogleMap
            mapContainerStyle={containerStyle}
            center={
              selectedListing
                ? { lat: selectedListing.lat, lng: selectedListing.lng }
                : defaultCenter
            }
            zoom={zoom}
            onLoad={(map) => {
              mapRef.current = map;
            }}
          >
            {/* Render markers for each listing */}
            {listings.map((listing) => (
              <Marker
                key={listing.id}
                position={{ lat: listing.lat, lng: listing.lng }}
                title={listing.title}
                icon={
                  selectedListing?.id === listing.id
                    ? "http://maps.google.com/mapfiles/ms/icons/blue-dot.png"
                    : undefined
                }
                onClick={() => handleClickListing(listing)}
              />
            ))}
          </GoogleMap>
        )}
      </div>
    </div>
  );
}