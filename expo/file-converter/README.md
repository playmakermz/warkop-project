# ⚡ File Converter — Expo React Native App

A simple Android file converter app built with **Expo + React Native**.
**No Android Studio required** — everything is done from the terminal.

## Supported Conversions

| From | To  | Works in Expo Go? | Description                     |
|------|-----|--------------------|---------------------------------|
| JPG  | PDF | ✅ Yes             | Convert images to PDF documents |
| MP4  | GIF | ⚙️ Needs EAS Build | Convert video to animated GIF   |
| MP3  | MP4 | ⚙️ Needs EAS Build | Convert audio to video file     |

All conversions happen **on-device** — your files never leave your phone.

---

## Prerequisites

You only need **two things** installed on your computer:

1. **Node.js** ≥ 18 → [Download](https://nodejs.org/)
2. **Expo Go** app on your Android phone → [Google Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)

That's it. No Android Studio. No Java. No Gradle.

---

## Quick Start (5 minutes)

### Step 1: Create a new Expo project

```bash
npx create-expo-app@latest FileConverter --template blank
cd FileConverter
```

### Step 2: Copy source files

Copy the contents of this project into your new Expo project:
- Replace `index.js` and `app.json`
- Copy the entire `src/` folder
- Replace `package.json` and `babel.config.js`

### Step 3: Install dependencies

```bash
npx expo install expo-document-picker expo-file-system expo-print expo-sharing expo-media-library expo-status-bar
npm install ffmpeg-kit-react-native
```

### Step 4: Test in Expo Go

```bash
npx expo start
```

Scan the QR code with the **Expo Go** app on your phone.
JPG → PDF conversion will work immediately in Expo Go!

> **Note:** MP4→GIF and MP3→MP4 use FFmpeg which requires a custom build.
> They'll show a helpful message in Expo Go pointing you to Step 5.

### Step 5: Build the APK (no Android Studio!)

Install EAS CLI (one-time):
```bash
npm install -g eas-cli
eas login
```

Build the APK in the cloud:
```bash
eas build --platform android --profile preview
```

EAS builds the APK **on Expo's cloud servers**. No local Android setup needed.
Once done, it gives you a download link for the `.apk` file.
Install it directly on your Android phone.

### Step 6 (Optional): Build for Google Play Store

```bash
eas build --platform android --profile production
```

This creates an `.aab` file you can upload to the Google Play Console.

---

## Project Structure

```
FileConverter/
├── index.js                  # Entry point (registers Expo app)
├── app.json                  # Expo config (permissions, plugins)
├── eas.json                  # EAS Build profiles
├── package.json              # Dependencies
├── babel.config.js           # Babel config for Expo
└── src/
    ├── App.js                # Root component with screen navigation
    ├── components/
    │   └── ConversionCard.js # Tappable card for each conversion type
    ├── screens/
    │   ├── HomeScreen.js     # Main menu with conversion options
    │   ├── ConvertScreen.js  # File picker + conversion progress
    │   └── ResultScreen.js   # Success screen with Share & Save
    └── utils/
        ├── constants.js      # Colors, conversion type definitions
        └── converter.js      # All conversion logic (expo-print, FFmpeg)
```

---

## How Each Conversion Works

### 🖼️ JPG → PDF
Uses **expo-print** to render the image inside HTML, then generates a PDF.
Works natively in Expo Go — no extra build needed.

### 🎬 MP4 → GIF
Uses **ffmpeg-kit-react-native** with a two-pass approach:
1. Generates an optimized color palette
2. Creates the GIF using that palette for high quality

Limited to 10 seconds, 480px wide, 10fps by default (configurable).

### 🎵 MP3 → MP4
Uses **ffmpeg-kit-react-native** to combine the audio with a
generated solid-color background video (1280×720).

---

## Commands Cheat Sheet

| Command                                             | What it does                    |
|-----------------------------------------------------|---------------------------------|
| `npx expo start`                                    | Start dev server (Expo Go)      |
| `eas build --platform android --profile preview`    | Build APK (cloud, no AS)        |
| `eas build --platform android --profile production` | Build AAB for Play Store        |
| `eas submit --platform android`                     | Submit to Google Play            |

---

## Troubleshooting

**"FFmpeg not available" error in Expo Go**
→ This is expected! MP4/MP3 conversions need a custom build.
→ Run: `eas build --platform android --profile preview`

**Document picker doesn't show files**
→ Make sure permissions are listed in `app.json`
→ On Android 13+, specific media permissions are needed (already configured)

**EAS build fails**
→ Run `eas login` first
→ Make sure `eas.json` is in the project root
→ Try `eas build --platform android --profile preview --clear-cache`

**Expo Go can't connect**
→ Make sure your phone and computer are on the same Wi-Fi
→ Try: `npx expo start --tunnel`

---

## No Android Studio. Ever.

This entire project is designed to be built without Android Studio:

| Task              | Tool                     |
|-------------------|--------------------------|
| Development       | Expo Go (phone app)      |
| Debug builds      | EAS Build (cloud)        |
| Release APK       | EAS Build (cloud)        |
| Play Store upload | EAS Submit (cloud)       |

---

## License

MIT — Free to use and modify.
