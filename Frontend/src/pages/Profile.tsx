import React, { useState, useRef, useEffect } from "react"
import { Autocomplete } from "@react-google-maps/api"
import { S3Client, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3"
import { getSignedUrl } from "@aws-sdk/s3-request-presigner"
import { useAppStore } from "../store/useAppStore"
import styles from "./Profile.module.css"

const PREFERENCES_API_URL = "http://localhost:3000/api/v1/preferences"
const PROFILE_API_URL = "http://localhost:3000/api/v1/profile"

const s3Client = new S3Client({
  region: import.meta.env.VITE_AWS_REGION || "us-east-1",
  credentials: {
    accessKeyId: import.meta.env.VITE_AWS_ACCESS_KEY_ID || "",
    secretAccessKey: import.meta.env.VITE_AWS_SECRET_ACCESS_KEY || "",
  },
})

type Preferences = {
  preference_id?: number
  user_id: number
  preferred_location: string
  budget_min: string
  budget_max: string
  preferred_roommates: string
  preferred_bednum: string
  preferred_bathnum: string
}

const makeEmptyPreferences = (userId: number): Preferences => ({
  user_id: userId,
  preferred_location: "",
  budget_min: "",
  budget_max: "",
  preferred_roommates: "",
  preferred_bednum: "",
  preferred_bathnum: "",
})

const parseNullableInt = (value: string): number | null => {
  if (value.trim() === "") return null
  const parsed = Number.parseInt(value, 10)
  return Number.isNaN(parsed) ? null : parsed
}

const parseNullableFloat = (value: string): number | null => {
  if (value.trim() === "") return null
  const parsed = Number.parseFloat(value)
  return Number.isNaN(parsed) ? null : parsed
}

const sanitizeForKey = (value: string) =>
  value
    .toLowerCase()
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9._-]/g, "")

const getS3PhotoUrl = async (
  bucketName: string,
  objectKey: string,
  expiresIn: number = 3600
): Promise<string> => {
  const command = new GetObjectCommand({
    Bucket: bucketName,
    Key: objectKey,
  })

  return getSignedUrl(s3Client, command, { expiresIn })
}

const uploadProfilePhotoToS3 = async (bucketName: string, objectKey: string, file: File) => {
  const fileBytes = new Uint8Array(await file.arrayBuffer())

  const command = new PutObjectCommand({
    Bucket: bucketName,
    Key: objectKey,
    Body: fileBytes,
    ContentType: file.type || "application/octet-stream",
  })

  await s3Client.send(command)
  return objectKey
}

type UserProfile = {
  user_id: number
  name?: string
  profile_photo?: string
}

const fetchProfile = async (userId: number): Promise<UserProfile | null> => {
  const response = await fetch(`${PROFILE_API_URL}?user_id=${userId}`)

  if (response.status === 404) {
    return null
  }

  const body = await response.json()

  if (!response.ok) {
    throw new Error(body?.error || `HTTP ${response.status}`)
  }

  return {
    user_id: body.user_id,
    name: body.name || "",
    profile_photo: body.profile_photo || "",
  }
}

const saveProfile = async (profile: UserProfile) => {
  const response = await fetch(PROFILE_API_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  })

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body?.error || `HTTP ${response.status}`)
  }
}

const fetchPreferences = async (userId: number): Promise<Preferences | null> => {
  const response = await fetch(`${PREFERENCES_API_URL}?user_id=${userId}`)

  if (response.status === 404) {
    return null
  }

  const body = await response.json()

  if (!response.ok) {
    throw new Error(body?.error || `HTTP ${response.status}`)
  }

  return {
    preference_id: body.preference_id,
    user_id: body.user_id,
    preferred_location: body.preferred_location || "",
    budget_min: body.budget_min?.toString() || "",
    budget_max: body.budget_max?.toString() || "",
    preferred_roommates: body.preferred_roommates?.toString() || "",
    preferred_bednum: body.preferred_bednum?.toString() || "",
    preferred_bathnum: body.preferred_bathnum?.toString() || "",
  }
}

const savePreferences = async (prefs: Preferences) => {
  const payload = {
    user_id: prefs.user_id,
    preferred_location: prefs.preferred_location.trim() || null,
    budget_min: parseNullableInt(prefs.budget_min),
    budget_max: parseNullableInt(prefs.budget_max),
    preferred_roommates: parseNullableInt(prefs.preferred_roommates),
    preferred_bednum: parseNullableInt(prefs.preferred_bednum),
    preferred_bathnum: parseNullableFloat(prefs.preferred_bathnum),
  }

  const response = await fetch(PREFERENCES_API_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(body?.error || `HTTP ${response.status}`)
  }
}

export default function Profile() {
  const userId = useAppStore((state) => state.userId)
  const [isEditing, setIsEditing] = useState(false)
  const [isLoadingPrefs, setIsLoadingPrefs] = useState(false)
  const [prefsStatusError, setPrefsStatusError] = useState("")
  const [prefsStatusSuccess, setPrefsStatusSuccess] = useState("")

  // Persisted profile state
  const [displayName, setDisplayName] = useState("John Doe")
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null)
  const [profilePhotoKey, setProfilePhotoKey] = useState<string>("")
  const [preferences, setPreferences] = useState<Preferences>(makeEmptyPreferences(userId ?? 0))

  // Temp state used while editing
  const [tempName, setTempName] = useState(displayName)
  const [tempAvatar, setTempAvatar] = useState<string | null>(null)
  const [tempAvatarFile, setTempAvatarFile] = useState<File | null>(null)
  const [tempPrefs, setTempPrefs] = useState<Preferences>(preferences)

  const fileRef = useRef<HTMLInputElement | null>(null)
  const locationAutocompleteRef = useRef<google.maps.places.Autocomplete | null>(null)

  useEffect(() => {
    return () => {
      if (tempAvatar) URL.revokeObjectURL(tempAvatar)
    }
  }, [tempAvatar])

  useEffect(() => {
    if (!userId) {
      setPrefsStatusError("Please log in to view your profile.")
      return
    }

    const loadProfileAndPreferences = async () => {
      setIsLoadingPrefs(true)
      setPrefsStatusError("")

      try {
        const [loadedProfile, loadedPreferences] = await Promise.all([
          fetchProfile(userId),
          fetchPreferences(userId),
        ])

        if (loadedProfile) {
          if (loadedProfile.name) {
            setDisplayName(loadedProfile.name)
            setTempName(loadedProfile.name)
          }

          if (loadedProfile.profile_photo) {
            setProfilePhotoKey(loadedProfile.profile_photo)

            const bucketName = import.meta.env.VITE_S3_BUCKET
            if (bucketName) {
              const signedProfileUrl = await getS3PhotoUrl(bucketName, loadedProfile.profile_photo)
              setAvatarUrl(signedProfileUrl)
            }
          }
        }

        if (loadedPreferences) {
          setPreferences(loadedPreferences)
          setTempPrefs(loadedPreferences)
        }
      } catch (error) {
        setPrefsStatusError((error as Error).message || "Failed to load preferences.")
      } finally {
        setIsLoadingPrefs(false)
      }
    }

    loadProfileAndPreferences()
  }, [userId])

  const startEdit = () => {
    setPrefsStatusError("")
    setPrefsStatusSuccess("")
    setTempName(displayName)
    setTempAvatar(null)
    setTempAvatarFile(null)
    setTempPrefs(preferences)
    setIsEditing(true)
  }

  const cancelEdit = () => {
    if (tempAvatar) {
      URL.revokeObjectURL(tempAvatar)
      setTempAvatar(null)
    }
    setTempAvatarFile(null)
    setIsEditing(false)
  }

  const saveEdit = async () => {
    if (
      tempPrefs.budget_min.trim() !== "" &&
      tempPrefs.budget_max.trim() !== "" &&
      Number(tempPrefs.budget_min) > Number(tempPrefs.budget_max)
    ) {
      setPrefsStatusError("Budget min cannot be greater than budget max.")
      return
    }

    setIsLoadingPrefs(true)
    setPrefsStatusError("")
    setPrefsStatusSuccess("")

    try {
      let nextProfilePhotoKey = profilePhotoKey

      if (tempAvatarFile) {
        const bucketName = import.meta.env.VITE_S3_BUCKET
        const accessKeyId = import.meta.env.VITE_AWS_ACCESS_KEY_ID
        const secretAccessKey = import.meta.env.VITE_AWS_SECRET_ACCESS_KEY

        if (!bucketName) {
          throw new Error("VITE_S3_BUCKET is not configured.")
        }

        if (!accessKeyId || !secretAccessKey) {
          throw new Error("AWS credentials are not configured for profile photo upload.")
        }

        const safeName = sanitizeForKey(tempName || "user")
        const safeFileName = sanitizeForKey(tempAvatarFile.name || "profile.jpg")
        const objectKey = `profiles/user-${userId}-${safeName}-${Date.now()}/${safeFileName}`

        nextProfilePhotoKey = await uploadProfilePhotoToS3(bucketName, objectKey, tempAvatarFile)
      }

      await Promise.all([
        saveProfile({
          user_id: userId ?? 0,
          name: tempName.trim(),
          profile_photo: nextProfilePhotoKey || undefined,
        }),
        savePreferences(tempPrefs),
      ])

      setDisplayName(tempName)
      if (tempAvatar) {
        setAvatarUrl(tempAvatar)
        setTempAvatar(null)
      }
      setTempAvatarFile(null)
      setProfilePhotoKey(nextProfilePhotoKey)
      setPreferences(tempPrefs)
      setIsEditing(false)
      setPrefsStatusSuccess("Profile and preferences saved.")
    } catch (error) {
      setPrefsStatusError((error as Error).message || "Failed to save profile and preferences.")
    } finally {
      setIsLoadingPrefs(false)
    }
  }

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files && e.target.files[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    if (tempAvatar) URL.revokeObjectURL(tempAvatar)
    setTempAvatar(url)
    setTempAvatarFile(f)
  }

  const onLocationLoad = (autocomplete: google.maps.places.Autocomplete) => {
    locationAutocompleteRef.current = autocomplete
  }

  const onLocationPlaceChanged = () => {
    if (!locationAutocompleteRef.current) return

    const place = locationAutocompleteRef.current.getPlace()
    if (!place.geometry || !place.formatted_address) return

    setTempPrefs((prev) => ({
      ...prev,
      preferred_location: place.formatted_address || "",
    }))
  }

  return (
    <div className={styles.profileContainer}>
      <div className={styles.headerRow}>
        <h1>Your Profile</h1>
        <div>
          {!isEditing ? (
            <button className={styles.editButton} onClick={startEdit} disabled={isLoadingPrefs}>
              Edit
            </button>
          ) : (
            <>
              <button className={styles.saveButton} onClick={saveEdit} disabled={isLoadingPrefs}>
                {isLoadingPrefs ? "Saving..." : "Save"}
              </button>
              <button className={styles.cancelButton} onClick={cancelEdit} disabled={isLoadingPrefs}>
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      {prefsStatusSuccess && <p className={styles.successText}>{prefsStatusSuccess}</p>}
      {prefsStatusError && <p className={styles.errorText}>{prefsStatusError}</p>}

      <div className={styles.profileCard}>
        <div className={styles.avatarSection}>
          <div className={styles.avatarWrap} onClick={() => fileRef.current?.click()}>
            {(!isEditing && avatarUrl) || tempAvatar ? (
              <img
                src={isEditing && tempAvatar ? tempAvatar : avatarUrl || undefined}
                alt="Profile"
                className={styles.avatar}
              />
            ) : (
              <div className={styles.initials}>
                {displayName.split(" ").map((s) => s[0]).slice(0, 2).join("")}
              </div>
            )}
            {isEditing && <div className={styles.avatarHint}>Click to change</div>}
          </div>

          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleAvatarChange}
          />

          <div className={styles.infoSection}>
            {!isEditing ? (
              <h2 className={styles.displayName}>{displayName}</h2>
            ) : (
              <input
                className={styles.nameInput}
                value={tempName}
                onChange={(e) => setTempName(e.target.value)}
              />
            )}
            <div className={styles.meta}>Member since Feb 2026</div>
          </div>
        </div>

        <div className={styles.prefsSection}>
          <h3>Preferences</h3>

          {!isEditing ? (
            <div className={styles.prefsView}>
              <div>
                <strong>Preferred Location:</strong> {preferences.preferred_location || "Not set"}
              </div>
              <div>
                <strong>Budget:</strong>{" "}
                {preferences.budget_min || preferences.budget_max
                  ? `${preferences.budget_min || "-"} to ${preferences.budget_max || "-"}`
                  : "Not set"}
              </div>
              <div>
                <strong>Preferred Roommates:</strong> {preferences.preferred_roommates || "Not set"}
              </div>
              <div>
                <strong>Preferred Beds:</strong> {preferences.preferred_bednum || "Not set"}
              </div>
              <div>
                <strong>Preferred Baths:</strong> {preferences.preferred_bathnum || "Not set"}
              </div>
            </div>
          ) : (
            <div className={styles.prefsEdit}>
              <label className={styles.fieldLabel}>
                Preferred location
                <Autocomplete
                  onLoad={onLocationLoad}
                  onPlaceChanged={onLocationPlaceChanged}
                  className={styles.locationAutocomplete}
                >
                  <input
                    value={tempPrefs.preferred_location}
                    onChange={(e) => setTempPrefs({ ...tempPrefs, preferred_location: e.target.value })}
                    placeholder="Enter an address"
                    className={styles.input}
                  />
                </Autocomplete>
              </label>

              <label className={styles.fieldLabel}>
                Budget min
                <input
                  type="number"
                  min="0"
                  value={tempPrefs.budget_min}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, budget_min: e.target.value })}
                  className={styles.input}
                />
              </label>

              <label className={styles.fieldLabel}>
                Budget max
                <input
                  type="number"
                  min="0"
                  value={tempPrefs.budget_max}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, budget_max: e.target.value })}
                  className={styles.input}
                />
              </label>

              <label className={styles.fieldLabel}>
                Preferred roommates
                <input
                  type="number"
                  min="0"
                  value={tempPrefs.preferred_roommates}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, preferred_roommates: e.target.value })}
                  className={styles.input}
                />
              </label>

              <label className={styles.fieldLabel}>
                Preferred bedrooms
                <input
                  type="number"
                  min="0"
                  value={tempPrefs.preferred_bednum}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, preferred_bednum: e.target.value })}
                  className={styles.input}
                />
              </label>

              <label className={styles.fieldLabel}>
                Preferred bathrooms
                <input
                  type="number"
                  min="0"
                  step="0.5"
                  value={tempPrefs.preferred_bathnum}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, preferred_bathnum: e.target.value })}
                  className={styles.input}
                />
              </label>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
