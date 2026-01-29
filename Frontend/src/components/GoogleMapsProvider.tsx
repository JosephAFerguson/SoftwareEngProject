// GoogleMapsProvider.tsx
import { useJsApiLoader } from "@react-google-maps/api";

const libraries: ("places")[] = ["places"];

export function GoogleMapsProvider({ children }: { children: React.ReactNode }) {
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;

  const { isLoaded, loadError } = useJsApiLoader({
    id: "google-maps",
    googleMapsApiKey: apiKey,
    libraries,
  });

  if (loadError) return <div>Error loading Google Maps</div>;
  if (!isLoaded) return <div>Loading Maps...</div>;

  return <>{children}</>;
}
