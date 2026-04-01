import { create } from "zustand"
import { persist } from "zustand/middleware"

export type AppState = {
  signedIn: boolean
  userId: number | null
  darkMode: boolean
  setSignedIn: (value: boolean) => void
  setUserId: (value: number | null) => void
  toggleDarkMode: () => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      signedIn: false,
      userId: null,
      darkMode: false,

      setSignedIn: (signedIn) => set({ signedIn }),
      setUserId: (userId) => set({ userId }),

      toggleDarkMode: () =>
        set((state) => ({ darkMode: !state.darkMode }))
    }),
    {
      name: "app-storage"
    }
  )
)
