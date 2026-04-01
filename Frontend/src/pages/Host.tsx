import React, { useRef, useState, useEffect } from "react";
import { Autocomplete } from "@react-google-maps/api";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { useAppStore } from "../store/useAppStore";
import styles from "./Host.module.css";

const API_URL = "http://localhost:3000/api/v1/rental";

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
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const [submitSuccess, setSubmitSuccess] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const autocompleteRef = useRef<google.maps.places.Autocomplete | null>(null);

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
  }, [userId]);

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
