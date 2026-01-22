import { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useAppStore } from "../store/useAppStore"
import styles from "./Signup.module.css"

export default function Signup() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const setSignedIn = useAppStore(state => state.setSignedIn)
  const navigate = useNavigate()

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()

    // TODO: Implement real signup logic & validation
    if (password !== confirmPassword) {
      alert("Passwords do not match!")
      return
    }

    //TODO send to backend

    // Update global state & redirect
    setSignedIn(true)
    navigate("/profile")
  }

  return (
    <div className={styles.signupContainer}>
      <h1 className={styles.header}>Sign Up</h1>

      <form className={styles.form} onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />

        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />

        <input
          type="password"
          placeholder="Confirm Password"
          value={confirmPassword}
          onChange={e => setConfirmPassword(e.target.value)}
          required
        />

        <button type="submit">Sign Up</button>
      </form>
    </div>
  )
}
