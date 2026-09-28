// lib/flag.ts
// Image-based flags via flagcdn.com — always render (unlike emoji on Windows).
// Maps team name → ISO 3166-1 alpha-2 code.

const CODES: Record<string, string> = {
    // CONCACAF
    "United States": "us", "USA": "us", "Mexico": "mx", "Canada": "ca", "Honduras": "hn",
    "Guatemala": "gt", "El Salvador": "sv", "Costa Rica": "cr", "Panama": "pa", "Jamaica": "jm",
    "Trinidad and Tobago": "tt", "Haiti": "ht", "Curacao": "cw", "Curaçao": "cw",
    // CONMEBOL
    "Brazil": "br", "Argentina": "ar", "Colombia": "co", "Uruguay": "uy", "Chile": "cl",
    "Ecuador": "ec", "Peru": "pe", "Venezuela": "ve", "Bolivia": "bo", "Paraguay": "py",
    // UEFA
    "France": "fr", "Germany": "de", "Spain": "es", "England": "gb-eng", "Portugal": "pt",
    "Netherlands": "nl", "Belgium": "be", "Italy": "it", "Croatia": "hr", "Serbia": "rs",
    "Switzerland": "ch", "Denmark": "dk", "Austria": "at", "Poland": "pl", "Ukraine": "ua",
    "Türkiye": "tr", "Turkiye": "tr", "Turkey": "tr", "Hungary": "hu", "Slovakia": "sk",
    "Czech Republic": "cz", "Czechia": "cz", "Romania": "ro", "Albania": "al", "Slovenia": "si",
    "Georgia": "ge", "Scotland": "gb-sct", "Wales": "gb-wls", "Norway": "no", "Sweden": "se",
    "Iceland": "is", "Finland": "fi", "Greece": "gr", "Ireland": "ie",
    // CAF
    "Morocco": "ma", "Senegal": "sn", "Nigeria": "ng", "Ghana": "gh", "Cameroon": "cm",
    "Ivory Coast": "ci", "Côte d'Ivoire": "ci", "Cote d'Ivoire": "ci", "Algeria": "dz",
    "Tunisia": "tn", "Egypt": "eg", "South Africa": "za", "Mali": "ml", "DR Congo": "cd", "Congo DR": "cd", "Bosnia-Herzegovina": "ba", "Bosnia and Herzegovina": "ba",
    "Democratic Republic of the Congo": "cd", "Zambia": "zm", "Guinea": "gn",
    "Cape Verde": "cv", "Burkina Faso": "bf", "Angola": "ao",
    // AFC
    "Japan": "jp", "South Korea": "kr", "Korea Republic": "kr", "Australia": "au", "Iran": "ir",
    "IR Iran": "ir", "Saudi Arabia": "sa", "Qatar": "qa", "Iraq": "iq", "Uzbekistan": "uz",
    "Jordan": "jo", "UAE": "ae", "United Arab Emirates": "ae", "Indonesia": "id",
    "New Zealand": "nz", "Oman": "om", "Bahrain": "bh", "China PR": "cn", "China": "cn",
}

export function flagUrl(teamName: string, size: "20" | "40" | "80" | "160" = "80"): string | null {
    const code = CODES[teamName]
    if (!code) return null
    return `https://flagcdn.com/w${size}/${code}.png`
}

export function flagCode(teamName: string): string | null {
    return CODES[teamName] ?? null
}

// Team primary colors — one bold, recognizable color per WC2026 team (kit/
// flag primary), used for the hero banner and other team-colored accents.
// Covers all 48 qualified teams (see backend/ml/wc_2026_config.py) plus a
// few common name aliases. Where a team's literal primary color is too low-
// contrast to use as a bold banner fill (England/white, Germany/black), a
// strong secondary associated with the team is used instead.
const TEAM_COLORS: Record<string, string> = {
    // Pot 1
    "Argentina": "#6cb4ee",
    "France": "#0055a4",
    "England": "#c8102e",
    "USA": "#3c3b6e", "United States": "#3c3b6e",
    "Spain": "#c60b1e",
    "Mexico": "#006847",
    "Germany": "#dd0000",
    "Netherlands": "#ff6600",
    "Canada": "#d80621",
    "Croatia": "#ff0000",
    "Italy": "#0066cc",
    "Morocco": "#c1272d",
    // Pot 2
    "Colombia": "#fcd116",
    "Uruguay": "#5cbfeb",
    "Japan": "#bc002d",
    "Brazil": "#fce803",
    "Senegal": "#00853f",
    "Portugal": "#006600",
    "South Korea": "#003478", "Korea Republic": "#003478",
    "Denmark": "#c60c30",
    "Belgium": "#fdda24",
    "Switzerland": "#da291c",
    "Austria": "#ed2939",
    "Ecuador": "#ffdd00",
    // Pot 3
    "Peru": "#d91023",
    "Iran": "#239f40", "IR Iran": "#239f40",
    "Australia": "#00843d",
    "Nigeria": "#008751",
    "Poland": "#dc143c",
    "Serbia": "#c6363c",
    "Turkey": "#e30a17", "Türkiye": "#e30a17", "Turkiye": "#e30a17",
    "Chile": "#d52b1e",
    "Ivory Coast": "#f77f00", "Côte d'Ivoire": "#f77f00", "Cote d'Ivoire": "#f77f00",
    "Egypt": "#ce1126",
    "Saudi Arabia": "#006c35",
    "Ghana": "#006b3f",
    // Pot 4
    "Venezuela": "#7b1113",
    "Algeria": "#006233",
    "South Africa": "#007a4d",
    "Qatar": "#8d1b3d",
    "Tunisia": "#e70013",
    "Paraguay": "#0038a8",
    "Panama": "#00205b",
    "Costa Rica": "#ce1126",
    "Wales": "#d30731",
    "Scotland": "#0065bd",
    "Honduras": "#0073cf",
    "Cameroon": "#007a33",

    // Remaining CODES entries not in the current placeholder 48-team draw
    // (backend/ml/wc_2026_config.py is explicitly marked as needing an
    // update once the real draw is known) — kept here so any team the live
    // worldcup26.ir feed actually returns still gets a real color instead
    // of falling through to the generic default.
    "Guatemala": "#4997d0",
    "El Salvador": "#0047ab",
    "Jamaica": "#007847",
    "Trinidad and Tobago": "#ce1126",
    "Haiti": "#00209f",
    "Curacao": "#002b7f", "Curaçao": "#002b7f",
    "Bolivia": "#d52b1e",
    "Ukraine": "#005bbb",
    "Hungary": "#ce2939",
    "Slovakia": "#0b4ea2",
    "Czech Republic": "#11457e", "Czechia": "#11457e",
    "Romania": "#00318f",
    "Albania": "#e41e20",
    "Slovenia": "#0a4595",
    "Georgia": "#e8112d",
    "Norway": "#ef2b2d",
    "Sweden": "#006aa7",
    "Iceland": "#02529c",
    "Finland": "#003580",
    "Greece": "#0d5eaf",
    "Ireland": "#169b62",
    "Mali": "#14b53a",
    "DR Congo": "#007fff", "Congo DR": "#007fff", "Bosnia-Herzegovina": "#002395", "Bosnia and Herzegovina": "#002395", "Democratic Republic of the Congo": "#007fff",
    "Zambia": "#198a00",
    "Guinea": "#ce1126",
    "Cape Verde": "#003893",
    "Burkina Faso": "#ef2b2d",
    "Angola": "#ce1126",
    "Iraq": "#ce1126",
    "Uzbekistan": "#0099b5",
    "Jordan": "#ce1126",
    "UAE": "#00732f", "United Arab Emirates": "#00732f",
    "Indonesia": "#ce1126",
    "New Zealand": "#00247d",
    "Oman": "#c8102e",
    "Bahrain": "#ce1126",
    "China PR": "#de2910", "China": "#de2910",
}
export function teamColor(teamName: string): string {
    return TEAM_COLORS[teamName] ?? "#4f86f7"
}