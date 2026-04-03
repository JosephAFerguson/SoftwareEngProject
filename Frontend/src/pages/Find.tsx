// ============================================================================
// IMPORTS
// ============================================================================

import { useRef, useState, useEffect } from "react";
import styles from './Find.module.css';
import { GoogleMap, Marker } from "@react-google-maps/api";
import { FiMinimize2 } from "react-icons/fi";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import pic1 from "../../public/unit-1/photo1.png";
import { useAppStore } from "../store/useAppStore";

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
  sqft?: number;
  roommates?: number;
  bednum?: number;
  bathnum?: number;
  pet_friendly?: boolean;
  available_from?: string;
  available_to?: string;
  photos?: string[];
  recommendation_score?: number;
};

type RentalApiResponse = {
  listing_id?: number;
  title?: string;
  price?: number;
  address?: string;
  sqft?: number;
  roommates?: number;
  bednum?: number;
  bathnum?: number;
  pet_friendly?: boolean;
  available_from?: string;
  available_to?: string;
  photos?: string[];
  recommendation_score?: number;
};

// ============================================================================
// CONSTANTS
// ============================================================================

/**
 * Backend API endpoint used for rental lookup
 */
const API_URL = "http://localhost:3000/api/v1/rental";
const RECOMMENDATION_API_URL = "http://localhost:3000/api/v1/recommendations";

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
// AWS S3 CONFIGURATION
// ============================================================================

/**
 * Initialize S3 client
 * Configure with your AWS credentials and region
 */
const s3Client = new S3Client({
  region: import.meta.env.VITE_AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: import.meta.env.VITE_AWS_ACCESS_KEY_ID || "",
    secretAccessKey: import.meta.env.VITE_AWS_SECRET_ACCESS_KEY || "",
  },
});

/**
 * Generate a presigned URL for an S3 object
 * @param bucketName - S3 bucket name
 * @param objectKey - Object key/path in S3
 * @param expiresIn - Expiration time in seconds (default: 3600 = 1 hour)
 * @returns Presigned URL for accessing the object, empty string if generation fails
 */
async function getS3PhotoUrl(
  bucketName: string,
  objectKey: string,
  expiresIn: number = 3600
): Promise<string> {
  try {
    // Check if credentials are configured
    const accessKeyId = import.meta.env.VITE_AWS_ACCESS_KEY_ID;
    const secretAccessKey = import.meta.env.VITE_AWS_SECRET_ACCESS_KEY;
    const region = import.meta.env.VITE_AWS_REGION;
    const bucket = import.meta.env.VITE_S3_BUCKET;

    console.log("[S3 DEBUG] Environment variables:", {
      accessKeyId: accessKeyId ? "SET" : "NOT SET",
      secretAccessKey: secretAccessKey ? "SET" : "NOT SET",
      region: region || "us-east-1",
      bucket: bucket || "sublease-photos"
    });
    
    if (!accessKeyId || !secretAccessKey) {
      console.warn("AWS credentials not configured. Photos will not load.");
      return "";
    }

    console.log(`[S3] Generating presigned URL for: s3://${bucketName}/${objectKey}`);

    const command = new GetObjectCommand({
      Bucket: bucketName,
      Key: objectKey,
    });

    const url = await getSignedUrl(s3Client, command, { expiresIn });
    console.log(`[S3] Successfully generated URL for ${objectKey}`);
    return url;
  } catch (error) {
    console.error("Error generating S3 URL:", error);
    return "";
  }
}

async function mapRentalToListing(rental: RentalApiResponse, index: number): Promise<Listing> {
  let photoUrls: string[] = [];

  if (rental.photos && Array.isArray(rental.photos)) {
    console.log(`[S3] Processing ${rental.photos.length} photos for listing ${rental.listing_id}:`, rental.photos);
    photoUrls = await Promise.all(
      rental.photos.map(async (photoKey: string) => {
        const bucketName = import.meta.env.VITE_S3_BUCKET || "sublease-photos";
        return await getS3PhotoUrl(bucketName, photoKey);
      })
    );

    photoUrls = photoUrls.filter((url) => url.length > 0);
    console.log(`[S3] Generated ${photoUrls.length} valid presigned URLs for listing ${rental.listing_id}`);
  }

  return {
    id: rental.listing_id || index + 1,
    title: rental.title || `Rental at ${rental.address}`,
    price: Number(rental.price || 0),
    address: rental.address || "",
    lat: 39.1310 + (Math.random() - 0.5) * 0.01,
    lng: -84.5165 + (Math.random() - 0.5) * 0.01,
    imageUrl: photoUrls.length > 0 ? photoUrls[0] : "",
    sqft: rental.sqft,
    roommates: rental.roommates,
    bednum: rental.bednum,
    bathnum: rental.bathnum,
    pet_friendly: rental.pet_friendly,
    available_from: rental.available_from,
    available_to: rental.available_to,
    photos: photoUrls,
    recommendation_score: rental.recommendation_score,
  };
}

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
  // ========================================================================
  // STATE
  // ========================================================================

  const signedIn = useAppStore((state) => state.signedIn);
  const userId = useAppStore((state) => state.userId);
  const [selectedListing, setSelectedListing] = useState<Listing | null>(null);
  const [lockedListingId, setLockedListingId] = useState<number | null>(null);
  const mapRef = useRef<google.maps.Map | null>(null);
  const [zoom, setZoom] = useState(14);

  const [allListings, setAllListings] = useState<Listing[]>([]);
  const [recommendedListingIds, setRecommendedListingIds] = useState<number[]>([]);
  const [listingPhotoIndices, setListingPhotoIndices] = useState<Map<number, number>>(new Map());

  // ========================================================================
  // EFFECTS
  // ========================================================================

  useEffect(() => {
    fetchAllListings();
  }, []);

  useEffect(() => {
    if (!signedIn || !userId) {
      setRecommendedListingIds([]);
      return;
    }

    fetchRecommendedListings(userId);
  }, [signedIn, userId]);

  // ========================================================================
  // API CALLS
  // ========================================================================

  const fetchAllListings = async () => {
    console.log("[DEBUG] fetchAllListings called");

    try {
      const response = await fetch(`${API_URL}/all`);
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log("[DEBUG] API response data:", data);
      const listings: Listing[] = await Promise.all(
        data.rentals.map((rental: RentalApiResponse, index: number) => mapRentalToListing(rental, index))
      );

      setAllListings(listings);
    } catch (error) {
      console.error("Error fetching listings:", error);
      setAllListings([]);
    }
  };

  const fetchRecommendedListings = async (activeUserId: number) => {
    try {
      const response = await fetch(
        `${RECOMMENDATION_API_URL}?user_id=${activeUserId}&top_n=5&persist=false`
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.error || `Recommendations unavailable (${response.status})`;
        console.warn(`Recommendation fetch failed: ${errorMsg}`);
        setRecommendedListingIds([]);
        return;
      }

      const data = await response.json();
      if (!data.recommendations || !Array.isArray(data.recommendations)) {
        console.warn("Recommendations response missing or invalid");
        setRecommendedListingIds([]);
        return;
      }

      const recommendationIds = Array.isArray(data.recommendations)
        ? data.recommendations
            .map((listing: RentalApiResponse) => listing.listing_id)
            .filter((listingId: number | undefined): listingId is number => typeof listingId === "number")
        : [];

      setRecommendedListingIds(recommendationIds);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setRecommendedListingIds([]);
    }
  };

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
   * Gets the current photo index for a specific listing
   */
  const getListingPhotoIndex = (listingId: number): number => {
    return listingPhotoIndices.get(listingId) || 0;
  };

  /**
   * Sets the current photo index for a specific listing
   */
  const setListingPhotoIndex = (listingId: number, index: number) => {
    setListingPhotoIndices(prev => new Map(prev.set(listingId, index)));
  };

  /**
   * Navigates to the next photo for a specific listing
   */
  const handleListingNextPhoto = (listingId: number, photos: string[]) => {
    const currentIndex = getListingPhotoIndex(listingId);
    const nextIndex = (currentIndex + 1) % photos.length;
    setListingPhotoIndex(listingId, nextIndex);
  };

  /**
   * Navigates to the previous photo for a specific listing
   */
  const handleListingPrevPhoto = (listingId: number, photos: string[]) => {
    const currentIndex = getListingPhotoIndex(listingId);
    const prevIndex = currentIndex === 0 ? photos.length - 1 : currentIndex - 1;
    setListingPhotoIndex(listingId, prevIndex);
  };

  const recommendedSet = new Set(recommendedListingIds);
  const uniqueListings = allListings.filter(
    (l, idx, arr) => arr.findIndex((x) => x.id === l.id) === idx
  );
  const displayedListings = [
    ...uniqueListings.filter((l) => recommendedSet.has(l.id)),
    ...uniqueListings.filter((l) => !recommendedSet.has(l.id)),
  ];

  // ========================================================================
  // RENDER
  // ========================================================================

  return (
    <div className={styles.findContainer}>
      {/* ============================================================== */}
      {/* LISTINGS SECTION (Left Column) */}
      {/* ============================================================== */}
      <div className={styles.listingsContainer}>
        {/* Listings grid */}
          <div
            className={`${styles.listingsGrid} ${
              lockedListingId !== null ? styles.hasExpanded : ""
            }`}
          >
          {/* Map through listings and render cards */}
          {displayedListings.map((listing) => (
            <div
              key={listing.id}
              id={`listing-${listing.id}`}
              className={`${styles.listingCard}
                ${recommendedListingIds.includes(listing.id) ? styles.recommended : ""}
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

                {/* Listing image with photo viewer */}
                {listing.photos && listing.photos.length > 0 ? (
                  <div className={styles.photoViewer}>
                    <div className={styles.photoContainer}>
                      {/* Previous photo arrow */}
                      {listing.photos.length > 1 && (
                        <button 
                          className={`${styles.photoArrow} ${styles.photoArrowLeft}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleListingPrevPhoto(listing.id, listing.photos!);
                          }}
                          title="Previous photo"
                        >
                          ‹
                        </button>
                      )}

                      {/* Main photo display */}
                      <img
                        src={listing.photos[getListingPhotoIndex(listing.id)] || pic1}
                        alt={`${listing.title} - Photo ${getListingPhotoIndex(listing.id) + 1}`}
                        className={styles.listingImage}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleListingNextPhoto(listing.id, listing.photos!);
                        }}
                        onError={(e) => {
                          console.log(`[IMAGE ERROR] Failed to load photo ${getListingPhotoIndex(listing.id) + 1} for listing ${listing.id}:`, listing.photos![getListingPhotoIndex(listing.id)]);
                          // Fallback to default image if photo fails to load
                          (e.target as HTMLImageElement).src = pic1;
                        }}
                      />

                      {/* Next photo arrow */}
                      {listing.photos.length > 1 && (
                        <button 
                          className={`${styles.photoArrow} ${styles.photoArrowRight}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleListingNextPhoto(listing.id, listing.photos!);
                          }}
                          title="Next photo"
                        >
                          ›
                        </button>
                      )}
                    </div>

                    {/* Photo indicators */}
                    {listing.photos.length > 1 && (
                      <div className={styles.photoIndicators}>
                        {listing.photos.map((_, index) => (
                          <span 
                            key={index}
                            className={`${styles.photoIndicator} ${index === getListingPhotoIndex(listing.id) ? styles.active : ''}`}
                            onClick={(e) => {
                              e.stopPropagation();
                              setListingPhotoIndex(listing.id, index);
                            }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <img
                    src={listing.imageUrl || pic1}
                    alt={listing.title}
                    className={styles.listingImage}
                    onError={(e) => {
                      console.log(`[IMAGE ERROR] Failed to load main image for listing ${listing.id}:`, listing.imageUrl);
                      // Fallback to default image if photo fails to load
                      (e.target as HTMLImageElement).src = pic1;
                    }}
                  />
                )}

                {/* Listing details */}
                <p>{listing.address}</p>
                <p className={styles.price}>${listing.price} / month</p>

                {/* Expanded details (only shown when card is maximized) */}
                {lockedListingId === listing.id && (
                  <div className={styles.expandedDetails}>
                    <h4>Listing Details</h4>
                
                    <div className={styles.detailsGrid}>
                      {listing.sqft && <p><strong>Square Footage:</strong> {listing.sqft} sq ft</p>}
                      {listing.roommates !== undefined && <p><strong>Roommates:</strong> {listing.roommates}</p>}
                      {listing.bednum !== undefined && <p><strong>Bedrooms:</strong> {listing.bednum}</p>}
                      {listing.bathnum !== undefined && <p><strong>Bathrooms:</strong> {listing.bathnum}</p>}
                      {listing.pet_friendly !== undefined && <p><strong>Pet Friendly:</strong> {listing.pet_friendly ? 'Yes' : 'No'}</p>}
                      {listing.available_from && <p><strong>Available From:</strong> {new Date(listing.available_from).toLocaleDateString()}</p>}
                      {listing.available_to && <p><strong>Available To:</strong> {new Date(listing.available_to).toLocaleDateString()}</p>}
                    </div>
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
          {displayedListings.map((listing) => (
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
      </div>
    </div>
  );
}