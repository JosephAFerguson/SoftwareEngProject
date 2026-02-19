import React, { useState, useRef, useEffect } from "react"
import styles from "./Profile.module.css"

type Preferences = {
  contactMethod: string
  allowPets: boolean
}

export default function Profile() {
  const [isEditing, setIsEditing] = useState(false)

  // Persisted profile state
  const [displayName, setDisplayName] = useState("John Doe")
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null)
  const [preferences, setPreferences] = useState<Preferences>({
    contactMethod: "Email",
    allowPets: true,
  })

  // Temp state used while editing
  const [tempName, setTempName] = useState(displayName)
  const [tempAvatar, setTempAvatar] = useState<string | null>(null)
  const [tempPrefs, setTempPrefs] = useState<Preferences>(preferences)

  const fileRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    return () => {
      if (tempAvatar) URL.revokeObjectURL(tempAvatar)
    }
  }, [tempAvatar])

  const startEdit = () => {
    setTempName(displayName)
    setTempAvatar(null)
    setTempPrefs(preferences)
    setIsEditing(true)
  }

  const cancelEdit = () => {
    if (tempAvatar) {
      URL.revokeObjectURL(tempAvatar)
      setTempAvatar(null)
    }
    setIsEditing(false)
  }

  const saveEdit = () => {
    setDisplayName(tempName)
    if (tempAvatar) {
      // revoke old avatar if any
      if (avatarUrl) URL.revokeObjectURL(avatarUrl)
      setAvatarUrl(tempAvatar)
      setTempAvatar(null)
    }
    setPreferences(tempPrefs)
    setIsEditing(false)
  }

  const handleAvatarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files && e.target.files[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    if (tempAvatar) URL.revokeObjectURL(tempAvatar)
    setTempAvatar(url)
  }

  return (
    <div className={styles.profileContainer}>
      <div className={styles.headerRow}>
        <h1>Your Profile</h1>
        <div>
          {!isEditing ? (
            <button className={styles.editButton} onClick={startEdit}>
              Edit
            </button>
          ) : (
            <>
              <button className={styles.saveButton} onClick={saveEdit}>
                Save
              </button>
              <button className={styles.cancelButton} onClick={cancelEdit}>
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

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
                <strong>Contact:</strong> {preferences.contactMethod}
              </div>
              <div>
                <strong>Pets allowed:</strong> {preferences.allowPets ? "Yes" : "No"}
              </div>
            </div>
          ) : (
            <div className={styles.prefsEdit}>
              <label className={styles.fieldLabel}>
                Contact method
                <input
                  value={tempPrefs.contactMethod}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, contactMethod: e.target.value })}
                  className={styles.input}
                />
              </label>

              <label className={styles.fieldLabelCheckbox}>
                <input
                  type="checkbox"
                  checked={tempPrefs.allowPets}
                  onChange={(e) => setTempPrefs({ ...tempPrefs, allowPets: e.target.checked })}
                />
                Allow pets
              </label>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
