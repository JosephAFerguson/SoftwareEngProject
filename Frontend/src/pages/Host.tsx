import React, { useRef, useState, useEffect } from "react";
import { Autocomplete } from "@react-google-maps/api";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { useAppStore } from "../store/useAppStore";
import styles from "./Host.module.css";

const API_URL = "http://localhost:3000/api/v1/rental";
const ALL_RENTALS_API_URL = "http://localhost:3000/api/v1/rental/all";

const s3UploadClient = new S3Client({
  region: import.meta.env.VITE_AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: import.meta.env.VITE_AWS_ACCESS_KEY_ID || "",
    secretAccessKey: import.meta.env.VITE_AWS_SECRET_ACCESS_KEY || "",
  },
});

const sanitizeForKey = (value: string) =>
  value
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9._-]/g, "");

const uploadPhotoToS3 = async (bucketName: string, objectKey: string, file: File) => {
  const fileBytes = new Uint8Array(await file.arrayBuffer());

  const command = new PutObjectCommand({
    Bucket: bucketName,
    Key: objectKey,
    Body: fileBytes,
    ContentType: file.type || "application/octet-stream",
  });

  await s3UploadClient.send(command);
  return objectKey;
};

const postRental = async (data: RentalData) => {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  const text = await response.text();

  let result;
  try {
    result = JSON.parse(text);
  } catch {
    result = text;
  }

  if (!response.ok) {
    throw new Error(
      typeof result === "string" ? result : result.error || `HTTP ${response.status}`
    );
  }

  return result;
};

const updateRental = async (data: ExistingListing) => {
  const response = await fetch(`${API_URL}/`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  const text = await response.text();

  let result;
  try {
    result = JSON.parse(text);
  } catch {
    result = text;
  }

  if (!response.ok) {
    if (response.status === 405) {
      throw new Error(
        "Listing edit endpoint is unavailable (405). Restart the backend server and make sure PUT /api/v1/rental is registered."
      );
    }

    const errorDetails =
      typeof result === "string"
        ? result
        : result.details || result.error || `HTTP ${response.status}`;
    throw new Error(`Update failed (${response.status}): ${errorDetails}`);
  }

  return result;
};

type RentalData = {
  user_id: number;
  title: string;
  address: string;
  price: number;
  sqft: number | null;
  roommates: number | null;
  bednum: number | null;
  bathnum: number | null;
  pet_friendly: boolean;
  available_from?: string;
  available_to?: string;
  photos?: string[];
};

type ExistingListing = {
  listing_id: number;
  user_id: number;
  title: string;
  address: string;
  price: number;
  sqft?: number;
  roommates?: number;
  bednum?: number;
  bathnum?: number;
  pet_friendly?: boolean;
  available_from?: string;
  available_to?: string;
  photos?: string[];
};

const makeNewRental = (userId: number): RentalData => ({
  user_id: userId,
  title: "",
  address: "123 Main St, Anytown, USA",
  price: 1200,
  sqft: 500,
  roommates: 2,
  bednum: 2,
  bathnum: 1,
  pet_friendly: false,
});

export default function Host() {
  const userId = useAppStore((state) => state.userId);
  const [newRentalData, setNewRentalData] = useState<RentalData>(makeNewRental(userId ?? 0));
  const [existingListing, setExistingListing] = useState<ExistingListing | null>(null);
  const [editingListing, setEditingListing] = useState<ExistingListing | null>(null);
  const [isEditingListing, setIsEditingListing] = useState(false);
  const [isLoadingListing, setIsLoadingListing] = useState(false);
  const [isSavingListing, setIsSavingListing] = useState(false);
  const [activePhotoIndex, setActivePhotoIndex] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [submitSuccess, setSubmitSuccess] = useState("");
  const [listingEditError, setListingEditError] = useState("");
  const [listingEditSuccess, setListingEditSuccess] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);
  const editAutocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);

  const fetchExistingListing = async (activeUserId: number) => {
    setIsLoadingListing(true);
    try {
      const response = await fetch(ALL_RENTALS_API_URL);
      if (!response.ok) {
        setExistingListing(null);
        return;
      }

      const data: { rentals?: ExistingListing[] } = await response.json();
      const listing = (data.rentals ?? []).find((r) => r.user_id === activeUserId) ?? null;
      setExistingListing(listing);
      setEditingListing(listing);
      setIsEditingListing(false);
      setActivePhotoIndex(0);
    } catch {
      setExistingListing(null);
    } finally {
      setIsLoadingListing(false);
    }
  };

  // Image Upload Handlers
  const handleUploadClick = () => fileInputRef.current?.click();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const fileArray = Array.from(files);
    // Append new files to existing files instead of replacing
    const updatedFiles = [...uploadedFiles, ...fileArray];
    setUploadedFiles(updatedFiles);

    // Create preview URLs for new files
    const newPreviewUrls = fileArray.map((file) => URL.createObjectURL(file));
    setPreviewUrls([...previewUrls, ...newPreviewUrls]);
    
    // Reset file input so same file can be selected again
    e.target.value = '';
  };

  const handleRemovePhoto = (idx: number) => {
    // Clean up the preview URL
    URL.revokeObjectURL(previewUrls[idx]);
    
    // Remove from both arrays
    const updatedFiles = uploadedFiles.filter((_, i) => i !== idx);
    const updatedUrls = previewUrls.filter((_, i) => i !== idx);
    
    setUploadedFiles(updatedFiles);
    setPreviewUrls(updatedUrls);
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    setSubmitError("");
    setSubmitSuccess("");

    if (!userId) {
      setSubmitError("Please log in before creating a listing.");
      return;
    }

    setIsSubmitting(true);

    try {
      const dataToSend: RentalData = {
        ...newRentalData,
        user_id: userId,
      };

      if (!dataToSend.available_from) delete dataToSend.available_from;
      if (!dataToSend.available_to) delete dataToSend.available_to;

      if (uploadedFiles.length > 0) {
        const bucketName = import.meta.env.VITE_S3_BUCKET;
        const accessKeyId = import.meta.env.VITE_AWS_ACCESS_KEY_ID;
        const secretAccessKey = import.meta.env.VITE_AWS_SECRET_ACCESS_KEY;

        if (!bucketName) {
          throw new Error("VITE_S3_BUCKET is not configured.");
        }

        if (!accessKeyId || !secretAccessKey) {
          throw new Error("AWS credentials are not configured for photo upload.");
        }

        const listingPrefix = sanitizeForKey(dataToSend.title || "listing");
        const uploadFolder = `listings/${listingPrefix}-${Date.now()}`;

        const uploadedPhotoKeys = await Promise.all(
          uploadedFiles.map((file, index) => {
            const safeFileName = sanitizeForKey(file.name || `photo-${index + 1}.jpg`);
            const objectKey = `${uploadFolder}/photo-${index + 1}-${safeFileName}`;
            return uploadPhotoToS3(bucketName, objectKey, file);
          })
        );

        dataToSend.photos = uploadedPhotoKeys;
      } else {
        delete dataToSend.photos;
      }

      await postRental(dataToSend);
      await fetchExistingListing(userId);
      window.dispatchEvent(new Event("listing-status-changed"));

      setSubmitSuccess("Listing created successfully.");
      setNewRentalData(makeNewRental(userId));

      previewUrls.forEach(URL.revokeObjectURL);
      setPreviewUrls([]);
      setUploadedFiles([]);
    } catch (error) {
      setSubmitError((error as Error).message || "Failed to create listing.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Autocomplete Handlers
  const onLoad = (autocomplete: google.maps.places.Autocomplete) => {
    autocompleteRef.current = autocomplete;
  };

  const onEditLoad = (autocomplete: google.maps.places.Autocomplete) => {
    editAutocompleteRef.current = autocomplete;
  };

  const onPlaceChanged = () => {
    if (!autocompleteRef.current) return;

    const place = autocompleteRef.current.getPlace();
    if (!place.geometry || !place.geometry.location) {
      alert("No location details found. Please try again.");
      return;
    }

    setNewRentalData((prev) => ({
      ...prev,
      address: place.formatted_address || "",
    }));
  };

  const onEditPlaceChanged = () => {
    if (!editAutocompleteRef.current) return;

    const place = editAutocompleteRef.current.getPlace();
    if (!place.geometry || !place.geometry.location) {
      alert("No location details found. Please try again.");
      return;
    }

    setEditingListing((prev) =>
      prev
        ? {
            ...prev,
            address: place.formatted_address || "",
          }
        : prev
    );
  };

  useEffect(() => {
    return () => {
      previewUrls.forEach(URL.revokeObjectURL);
    };
  }, [previewUrls]);

  useEffect(() => {
    if (!userId) return;

    setNewRentalData((prev) => ({
      ...prev,
      user_id: userId,
    }));

    fetchExistingListing(userId);
  }, [userId]);

  const listingPhotos = existingListing?.photos ?? [];

  const handleStartEditingListing = () => {
    if (!existingListing) return;

    setEditingListing({ ...existingListing });
    setIsEditingListing(true);
    setListingEditError("");
    setListingEditSuccess("");
  };

  const handleCancelEditingListing = () => {
    setEditingListing(existingListing ? { ...existingListing } : null);
    setIsEditingListing(false);
    setListingEditError("");
    setListingEditSuccess("");
  };

  const handleSaveListingChanges = async () => {
    if (!editingListing || !userId) return;

    setListingEditError("");
    setListingEditSuccess("");
    setIsSavingListing(true);

    try {
      const payload: ExistingListing = {
        ...editingListing,
        user_id: userId,
      };

      if (!payload.available_from) delete payload.available_from;
      if (!payload.available_to) delete payload.available_to;

      await updateRental(payload);
      await fetchExistingListing(userId);
      setIsEditingListing(false);
      setListingEditSuccess("Listing updated successfully.");
    } catch (error) {
      setListingEditError((error as Error).message || "Failed to update listing.");
    } finally {
      setIsSavingListing(false);
    }
  };

  if (!userId) {
    return (
      <div className={styles.hostContainer}>
        <div className={styles.hostInputs}>
          <h2>Create a Listing</h2>
          <p className={styles.errorText}>Please log in before creating or viewing your listing.</p>
        </div>
      </div>
    );
  }

  if (isLoadingListing) {
    return (
      <div className={styles.fullscreenContainer}>
        <div className={styles.fullscreenCard}>
          <h2>Your Listing</h2>
          <p>Loading listing...</p>
        </div>
      </div>
    );
  }

  if (existingListing) {
    const isEditing =
      isEditingListing && !!editingListing && editingListing.listing_id === existingListing.listing_id;

    return (
      <div className={styles.fullscreenContainer}>
        <div className={styles.fullscreenCard}>
          <div className={styles.fullscreenHeader}>
            <h2>Your Listing</h2>
            <div className={styles.fullscreenHeaderActions}>
              <button
                type="button"
                className={styles.submitButton}
                onClick={handleStartEditingListing}
                disabled={isSavingListing}
              >
                Edit Listing
              </button>
            </div>
          </div>
          {listingEditSuccess && <p className={styles.successText}>{listingEditSuccess}</p>}
          {listingEditError && <p className={styles.errorText}>{listingEditError}</p>}

          {isEditing && editingListing ? (
            <div className={styles.fullscreenEditForm}>
              <div className={styles.formGroup}>
                <label htmlFor="edit-title">Title</label>
                <input
                  id="edit-title"
                  type="text"
                  value={editingListing.title}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, title: e.target.value } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-address">Address</label>
                <Autocomplete onLoad={onEditLoad} onPlaceChanged={onEditPlaceChanged}>
                  <input
                    id="edit-address"
                    type="text"
                    value={editingListing.address}
                    onChange={(e) =>
                      setEditingListing((prev) =>
                        prev ? { ...prev, address: e.target.value } : prev
                      )
                    }
                    className={styles.autocompleteInput}
                    style={{ width: "100%", boxSizing: "border-box" }}
                  />
                </Autocomplete>
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-price">Price ($/month)</label>
                <input
                  id="edit-price"
                  type="number"
                  min="1"
                  value={editingListing.price}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, price: Number(e.target.value) || 0 } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-sqft">Square Footage</label>
                <input
                  id="edit-sqft"
                  type="number"
                  value={editingListing.sqft ?? ""}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, sqft: e.target.value ? Number(e.target.value) : undefined } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-roommates">Number of Roommates</label>
                <input
                  id="edit-roommates"
                  type="number"
                  value={editingListing.roommates ?? ""}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, roommates: e.target.value ? Number(e.target.value) : undefined } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-bednum">Number of Beds</label>
                <input
                  id="edit-bednum"
                  type="number"
                  value={editingListing.bednum ?? ""}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, bednum: e.target.value ? Number(e.target.value) : undefined } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-bathnum">Number of Baths</label>
                <input
                  id="edit-bathnum"
                  type="number"
                  step="0.5"
                  value={editingListing.bathnum ?? ""}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, bathnum: e.target.value ? Number(e.target.value) : undefined } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-pet-friendly">
                  <input
                    id="edit-pet-friendly"
                    type="checkbox"
                    checked={Boolean(editingListing.pet_friendly)}
                    onChange={(e) =>
                      setEditingListing((prev) =>
                        prev ? { ...prev, pet_friendly: e.target.checked } : prev
                      )
                    }
                  />
                  Pet Friendly
                </label>
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-available-from">Available From</label>
                <input
                  id="edit-available-from"
                  type="date"
                  value={editingListing.available_from ?? ""}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, available_from: e.target.value || undefined } : prev
                    )
                  }
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit-available-to">Available To</label>
                <input
                  id="edit-available-to"
                  type="date"
                  value={editingListing.available_to ?? ""}
                  onChange={(e) =>
                    setEditingListing((prev) =>
                      prev ? { ...prev, available_to: e.target.value || undefined } : prev
                    )
                  }
                />
              </div>

              <div className={styles.fullscreenEditActions}>
                <button
                  type="button"
                  className={styles.submitButton}
                  onClick={handleSaveListingChanges}
                  disabled={isSavingListing}
                >
                  {isSavingListing ? "Saving..." : "Save Changes"}
                </button>

                <button
                  type="button"
                  className={styles.cancelButton}
                  onClick={handleCancelEditingListing}
                  disabled={isSavingListing}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              <p className={styles.fullscreenSubtitle}>{existingListing.title}</p>
              <p>{existingListing.address}</p>
            </>
          )}

          {listingPhotos.length > 0 ? (
            <div className={styles.fullscreenPhotoContainer}>
              {listingPhotos.length > 1 && (
                <button
                  type="button"
                  className={styles.fullscreenPhotoNav}
                  onClick={() =>
                    setActivePhotoIndex((prev) =>
                      prev === 0 ? listingPhotos.length - 1 : prev - 1
                    )
                  }
                >
                  ‹
                </button>
              )}

              <img
                src={listingPhotos[activePhotoIndex]}
                alt={existingListing.title}
                className={styles.fullscreenPhoto}
              />

              {listingPhotos.length > 1 && (
                <button
                  type="button"
                  className={styles.fullscreenPhotoNav}
                  onClick={() =>
                    setActivePhotoIndex((prev) =>
                      prev === listingPhotos.length - 1 ? 0 : prev + 1
                    )
                  }
                >
                  ›
                </button>
              )}
            </div>
          ) : (
            <div className={styles.noPhoto}>No photos uploaded.</div>
          )}

          <div className={styles.fullscreenDetails}>
            <p><strong>Price:</strong> ${existingListing.price} / month</p>
            {existingListing.sqft !== undefined && <p><strong>Square Footage:</strong> {existingListing.sqft} sq ft</p>}
            {existingListing.roommates !== undefined && <p><strong>Roommates:</strong> {existingListing.roommates}</p>}
            {existingListing.bednum !== undefined && <p><strong>Bedrooms:</strong> {existingListing.bednum}</p>}
            {existingListing.bathnum !== undefined && <p><strong>Bathrooms:</strong> {existingListing.bathnum}</p>}
            {existingListing.pet_friendly !== undefined && (
              <p><strong>Pet Friendly:</strong> {existingListing.pet_friendly ? "Yes" : "No"}</p>
            )}
            {existingListing.available_from && (
              <p><strong>Available From:</strong> {new Date(existingListing.available_from).toLocaleDateString()}</p>
            )}
            {existingListing.available_to && (
              <p><strong>Available To:</strong> {new Date(existingListing.available_to).toLocaleDateString()}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.hostContainer}>
      <div className={styles.hostInputs}>
        <h2>Create a Listing</h2>
        {submitSuccess && <p className={styles.successText}>{submitSuccess}</p>}
        {submitError && <p className={styles.errorText}>{submitError}</p>}

        <form className={styles.form} onSubmit={handleFormSubmit}>
          <div className={styles.formGroup}>
            <label htmlFor="title">Title</label>
            <input
              id="title"
              type="text"
              required
              value={newRentalData.title}
              onChange={(e) =>
                setNewRentalData((prev) => ({ ...prev, title: e.target.value }))
              }
              placeholder="Cozy room near campus"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="location">Location</label>
            <Autocomplete
              onLoad={onLoad}
              onPlaceChanged={onPlaceChanged}
            >
              <input
                type="text"
                required
                value={newRentalData.address}
                onChange={(e) =>
                  setNewRentalData((prev) => ({ ...prev, address: e.target.value }))
                }
                placeholder="Enter an address"
                className={styles.autocompleteInput}
                style={{
                  width: "100%",
                  padding: "10px",
                  fontSize: "16px",
                  borderRadius: "4px",
                  border: "1px solid #ccc",
                }}
              />
            </Autocomplete>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="price">Price ($/month)</label>
            <input
              id="price"
              type="number"
              min="1"
              required
              value={newRentalData.price}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  price: Number(e.target.value) || 0,
                }))
              }
              placeholder="1200"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="sqft">Square Footage</label>
            <input
              id="sqft"
              type="number"
              value={newRentalData.sqft || ""}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  sqft: e.target.value ? Number(e.target.value) : null,
                }))
              }
              placeholder="500"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="roommates">Number of Roommates</label>
            <input
              id="roommates"
              type="number"
              value={newRentalData.roommates || ""}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  roommates: e.target.value ? Number(e.target.value) : null,
                }))
              }
              placeholder="2"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="bednum">Number of Beds</label>
            <input
              id="bednum"
              type="number"
              value={newRentalData.bednum || ""}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  bednum: e.target.value ? Number(e.target.value) : null,
                }))
              }
              placeholder="2"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="bathnum">Number of Baths</label>
            <input
              id="bathnum"
              type="number"
              step="0.5"
              value={newRentalData.bathnum || ""}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  bathnum: e.target.value ? Number(e.target.value) : null,
                }))
              }
              placeholder="1"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="pet_friendly">
              <input
                id="pet_friendly"
                type="checkbox"
                checked={newRentalData.pet_friendly}
                onChange={(e) =>
                  setNewRentalData((prev) => ({
                    ...prev,
                    pet_friendly: e.target.checked,
                  }))
                }
              />
              Pet Friendly
            </label>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="available_from">Available From</label>
            <input
              id="available_from"
              type="date"
              value={newRentalData.available_from || ""}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  available_from: e.target.value,
                }))
              }
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="available_to">Available To</label>
            <input
              id="available_to"
              type="date"
              value={newRentalData.available_to || ""}
              onChange={(e) =>
                setNewRentalData((prev) => ({
                  ...prev,
                  available_to: e.target.value,
                }))
              }
            />
          </div>

          <button type="submit" className={styles.submitButton} disabled={isSubmitting}>
            {isSubmitting ? "Creating Listing..." : "Create Listing"}
          </button>
        </form>
      </div>

      <div className={styles.hostPictureUpload}>
        <div
          className={`${styles.uploadBox} ${previewUrls.length > 0 ? styles.hasImages : ''}`}
          onClick={handleUploadClick}
        >
          {previewUrls.length > 0 && (
            <div className={styles.previewContainer}>
              {previewUrls.map((url, idx) => (
                <div key={idx} className={styles.photoWrapper}>
                  <img
                    src={url}
                    alt="Preview"
                    className={styles.previewImg}
                  />
                  <button
                    type="button"
                    className={styles.deleteButton}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemovePhoto(idx);
                    }}
                    title="Remove photo"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}

          <p>Upload Photos</p>
          <span>Click to select images</span>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className={styles.hiddenFileInput}
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
      </div>
    </div>
  );
}
