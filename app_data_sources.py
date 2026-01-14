"""
Static databases and LLM-powered data generation for apps.

This module provides:
1. Static databases for apps that need global data (Spotify songs, etc.)
2. LLM-powered generators for personalized user data
"""

from typing import Any, Dict, List, Optional
import random


# =============================================================================
# Static Databases
# =============================================================================

class SpotifyDatabase:
    """Static database of songs for Spotify app."""

    # Small static database of songs
    SONGS_DB = [
        {
            "song_id": "track_indie_001",
            "title": "Midnight City",
            "artist": "M83",
            "album": "Hurry Up, We're Dreaming",
            "duration": 244,
            "genre": "Electronic"
        },
        {
            "song_id": "track_indie_002",
            "title": "Electric Feel",
            "artist": "MGMT",
            "album": "Oracular Spectacular",
            "duration": 229,
            "genre": "Indie Rock"
        },
        {
            "song_id": "track_indie_003",
            "title": "Pumped Up Kicks",
            "artist": "Foster the People",
            "album": "Torches",
            "duration": 239,
            "genre": "Indie Pop"
        },
        {
            "song_id": "track_rock_001",
            "title": "Seven Nation Army",
            "artist": "The White Stripes",
            "album": "Elephant",
            "duration": 231,
            "genre": "Rock"
        },
        {
            "song_id": "track_electronic_001",
            "title": "Strobe",
            "artist": "deadmau5",
            "album": "For Lack of a Better Name",
            "duration": 644,
            "genre": "Electronic"
        },
        {
            "song_id": "track_pop_001",
            "title": "Blinding Lights",
            "artist": "The Weeknd",
            "album": "After Hours",
            "duration": 200,
            "genre": "Pop"
        },
        {
            "song_id": "track_pop_002",
            "title": "Levitating",
            "artist": "Dua Lipa",
            "album": "Future Nostalgia",
            "duration": 203,
            "genre": "Pop"
        },
        {
            "song_id": "track_workout_001",
            "title": "Till I Collapse",
            "artist": "Eminem",
            "album": "The Eminem Show",
            "duration": 297,
            "genre": "Hip Hop"
        },
        {
            "song_id": "track_workout_002",
            "title": "Stronger",
            "artist": "Kanye West",
            "album": "Graduation",
            "duration": 311,
            "genre": "Hip Hop"
        },
        {
            "song_id": "track_chill_001",
            "title": "Weightless",
            "artist": "Marconi Union",
            "album": "Weightless",
            "duration": 477,
            "genre": "Ambient"
        }
    ]

    @classmethod
    def search_songs(cls, query: str, limit: int = 10) -> List[Dict]:
        """Search for songs matching query."""
        query_lower = query.lower()
        results = []

        for song in cls.SONGS_DB:
            # Match against title, artist, album, or genre
            if (query_lower in song["title"].lower() or
                query_lower in song["artist"].lower() or
                query_lower in song["album"].lower() or
                query_lower in song["genre"].lower()):
                results.append(song.copy())

        # If no matches, return random songs
        if not results:
            results = random.sample(cls.SONGS_DB, min(limit, len(cls.SONGS_DB)))

        return results[:limit]

    @classmethod
    def get_song_by_id(cls, song_id: str) -> Optional[Dict]:
        """Get song by ID."""
        for song in cls.SONGS_DB:
            if song["song_id"] == song_id:
                return song.copy()
        return None

    @classmethod
    def get_random_song(cls) -> Dict:
        """Get a random song."""
        return random.choice(cls.SONGS_DB).copy()

    @classmethod
    def get_songs_by_genre(cls, genre: str, limit: int = 5) -> List[Dict]:
        """Get songs by genre."""
        matching = [s for s in cls.SONGS_DB if s["genre"].lower() == genre.lower()]
        if matching:
            return random.sample(matching, min(limit, len(matching)))
        return random.sample(cls.SONGS_DB, min(limit, len(cls.SONGS_DB)))

    @classmethod
    def get_songs_by_artist(cls, artist: str, limit: int = 5) -> List[Dict]:
        """Get songs by artist."""
        matching = [s for s in cls.SONGS_DB if s["artist"].lower() == artist.lower()]
        if matching:
            return random.sample(matching, min(limit, len(matching)))
        return []


# =============================================================================
# LLM-Powered Generators
# =============================================================================

class LLMDataGenerator:
    """
    LLM-powered data generator for personalized user data.

    This generates realistic, contextual data based on user profile and description.
    """

    def __init__(self, llm_client=None):
        """
        Initialize with optional LLM client.

        Args:
            llm_client: Optional LLMClient for generating data.
                       If None, falls back to rule-based generation.
        """
        self.llm_client = llm_client

    def generate_fitness_baseline(
        self,
        user_profile: Dict[str, Any],
        description: str
    ) -> Dict[str, Any]:
        """
        Generate user fitness baseline.

        For first-time fitness app use, generates realistic baseline metrics.
        """
        if self.llm_client:
            return self._generate_fitness_baseline_with_llm(user_profile, description)
        else:
            return self._generate_fitness_baseline_rule_based(user_profile, description)

    def _generate_fitness_baseline_with_llm(
        self,
        user_profile: Dict[str, Any],
        description: str
    ) -> Dict[str, Any]:
        """Generate fitness baseline using LLM."""
        # For now, fall back to rule-based
        # TODO: Implement LLM-based generation
        return self._generate_fitness_baseline_rule_based(user_profile, description)

    def _generate_fitness_baseline_rule_based(
        self,
        user_profile: Dict[str, Any],
        description: str
    ) -> Dict[str, Any]:
        """Generate fitness baseline using rules."""
        # Extract age if available
        age = 28  # default

        # Generate resting heart rate based on age and fitness level
        desc_lower = description.lower()
        if "athlete" in desc_lower or "advanced" in desc_lower:
            fitness_level = "advanced"
            resting_hr = random.randint(45, 60)
        elif "beginner" in desc_lower or "start" in desc_lower:
            fitness_level = "beginner"
            resting_hr = random.randint(70, 85)
        else:
            fitness_level = "intermediate"
            resting_hr = random.randint(60, 70)

        return {
            "avg_resting_heart_rate": resting_hr,
            "fitness_level": fitness_level,
            "age_group": f"{(age // 10) * 10}-{(age // 10) * 10 + 10}",
            "weight_kg": random.randint(60, 90),
            "height_cm": random.randint(160, 190)
        }

    def generate_llm_chat_response(
        self,
        user_message: str,
        conversation_history: List[Dict],
        user_profile: Dict[str, Any]
    ) -> str:
        """
        Generate LLM chat response.

        Args:
            user_message: User's message
            conversation_history: Previous conversation history
            user_profile: User profile for context

        Returns:
            AI response string
        """
        if self.llm_client:
            return self._generate_response_with_llm(
                user_message,
                conversation_history,
                user_profile
            )
        else:
            return self._generate_response_rule_based(user_message)

    def _generate_response_with_llm(
        self,
        user_message: str,
        conversation_history: List[Dict],
        user_profile: Dict[str, Any]
    ) -> str:
        """Generate response using LLM."""
        # TODO: Implement actual LLM call
        # For now, fall back to rule-based
        return self._generate_response_rule_based(user_message)

    def _generate_response_rule_based(self, user_message: str) -> str:
        """Generate response using simple rules."""
        msg_lower = user_message.lower()

        if "?" in user_message:
            # Question
            if "how" in msg_lower:
                return "Here's how you can approach that: [AI provides step-by-step guidance]"
            elif "why" in msg_lower:
                return "The reason is: [AI explains the rationale and background]"
            elif "what" in msg_lower:
                return "What you're asking about is: [AI provides definition and context]"
            elif "when" in msg_lower:
                return "The best time would be: [AI provides timing recommendations]"
            else:
                return "Based on your question, here's what I can tell you: [AI provides helpful information]"
        elif "recommend" in msg_lower or "suggest" in msg_lower:
            return "I'd recommend: [AI provides personalized recommendations based on your goals]"
        elif "help" in msg_lower:
            return "I can help you with that. [AI provides assistance and next steps]"
        else:
            return "I understand. Let me help you with that. [AI provides relevant guidance and support]"


# =============================================================================
# Global Instances
# =============================================================================

# Singleton data generator (can be initialized with LLM client later)
_global_data_generator: Optional[LLMDataGenerator] = None


def get_data_generator(llm_client=None) -> LLMDataGenerator:
    """Get or create global data generator."""
    global _global_data_generator
    if _global_data_generator is None:
        _global_data_generator = LLMDataGenerator(llm_client)
    elif llm_client is not None and _global_data_generator.llm_client is None:
        # Update with LLM client if provided
        _global_data_generator.llm_client = llm_client
    return _global_data_generator


def set_llm_client_for_data_generation(llm_client):
    """Set LLM client for data generation."""
    global _global_data_generator
    if _global_data_generator is None:
        _global_data_generator = LLMDataGenerator(llm_client)
    else:
        _global_data_generator.llm_client = llm_client
