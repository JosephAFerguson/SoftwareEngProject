import React, { useRef, useState, useEffect } from "react";
import { Autocomplete } from "@react-google-maps/api";
import styles from "./Host.module.css";

const libraries: ("places")[] = ["places"];


export default function Host() {
  const [address, setAddress] = useState("");
  const [coordinates, setCoordinates] = useState({ lat: 0, lng: 0 });
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);

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

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Submitting:", { address, coordinates, uploadedFiles });
    // Add your submission logic here
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

    setAddress(place.formatted_address || "");
    setCoordinates({
      lat: place.geometry.location.lat(),
      lng: place.geometry.location.lng(),
    });
  };

  useEffect(() => {
    return () => {
      previewUrls.forEach(URL.revokeObjectURL);
    };
  }, [previewUrls]);

  return (
    <div className={styles.hostContainer}>
      <div className={styles.hostInputs}>
        <h2>Create a Listing</h2>

        <form className={styles.form} onSubmit={handleFormSubmit}>
          <div className={styles.formGroup}>
            <label htmlFor="title">Title</label>
            <input id="title" type="text" placeholder="Cozy room near campus" />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="location">Location</label>
            <Autocomplete
              onLoad={onLoad}
              onPlaceChanged={onPlaceChanged}
            >
              <input
                type="text"
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
            <input type="hidden" name="address" value={address} />
          </div>

          <button type="submit" className={styles.submitButton}>
            Create Listing
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
