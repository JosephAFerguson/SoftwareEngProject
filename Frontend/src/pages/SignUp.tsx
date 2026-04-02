import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./SignUp.module.css"

const SIGNUP_API_URL = "http://localhost:3000/api/v1/auth/signup"
const UC_EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@(mail\.uc\.edu|uc\.edu)$/

export default function Signup() {
  const [name, setName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")

  const setSignedIn = useAppStore(state => state.setSignedIn)
  const setUserId = useAppStore(state => state.setUserId)
  const navigate = useNavigate()

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setErrorMessage("")

    if (!name.trim()) {
      setErrorMessage("Please enter your full name.")
      return
    }

    if (!UC_EMAIL_REGEX.test(email.trim())) {
      setErrorMessage("Please use your UC email address (@mail.uc.edu or @uc.edu).")
      return
    }

    if (password.length < 8) {
      setErrorMessage("Password must be at least 8 characters.")
      return
    }

    if (password !== confirmPassword) {
      setErrorMessage("Passwords do not match.")
      return
    }

    setIsSubmitting(true)
    try {
      const response = await fetch(SIGNUP_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          email: email.trim().toLowerCase(),
          password,
        }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body?.error || "Failed to sign up.")
      }
      const body = await response.json().catch(() => ({}))
      const userId = typeof body?.user_id === "number" ? body.user_id : null
      if (!userId) {
        throw new Error("Signup succeeded but no user_id was returned.")
      }
      setSignedIn(true)
      setUserId(userId)
      navigate("/profile")
    } catch (error) {
      setErrorMessage((error as Error).message || "Failed to sign up.")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className={styles.signupContainer}>
      <div className={styles.card}>
        <div className={styles.logoRow}>
          <span className={styles.logoIcon}>🏠</span>
          <span className={styles.logoText}>SubleasesInc</span>
        </div>
        <h1 className={styles.header}>Create account</h1>
        <p className={styles.subheader}>UC students only — use your university email</p>

        {errorMessage && <p className={styles.errorText}>{errorMessage}</p>}

        <form className={styles.form} onSubmit={handleSubmit}>
          <div className={styles.fieldGroup}>
            <label className={styles.label}>Full Name</label>
            <input
              type="text"
              placeholder="Jane Doe"
              value={name}
              onChange={e => setName(e.target.value)}
              className={styles.input}
              required
            />
          </div>

          <div className={styles.fieldGroup}>
            <label className={styles.label}>UC Email</label>
            <input
              type="email"
              placeholder="m12345@mail.uc.edu"
              value={email}
              onChange={e => setEmail(e.target.value)}
              className={styles.input}
              required
            />
          </div>

          <div className={styles.fieldGroup}>
            <label className={styles.label}>Password</label>
            <input
              type="password"
              placeholder="Min. 8 characters"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className={styles.input}
              required
            />
          </div>

          <div className={styles.fieldGroup}>
            <label className={styles.label}>Confirm Password</label>
            <input
              type="password"
              placeholder="Re-enter password"
              value={confirmPassword}
              onChange={e => setConfirmPassword(e.target.value)}
              className={styles.input}
              required
            />
          </div>

          <button type="submit" disabled={isSubmitting} className={styles.submitBtn}>
            {isSubmitting ? "Creating Account..." : "Sign Up"}
          </button>
        </form>

        <p className={styles.switchText}>
          Already have an account? <a href="/login" className={styles.link}>Log in</a>
        </p>
      </div>
    </div>
  )
}