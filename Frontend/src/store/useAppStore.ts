import { create } from "zustand"
import { persist } from "zustand/middleware"

export type AppState = {
  signedIn: boolean
  darkMode: boolean
  setSignedIn: (value: boolean) => void
  toggleDarkMode: () => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      signedIn: false,
      darkMode: false,

      setSignedIn: (signedIn) => set({ signedIn }),

      toggleDarkMode: () =>
        set((state) => ({ darkMode: !state.darkMode }))
    }),
    {
      name: "app-storage"
    }
  )
)
