"""
App System for generating realistic app logs with stateful consistency.

Each app maintains state about the user to ensure consistency across:
- Different domains calling the same app
- Multiple calls within the same domain
"""

from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from mem_bench.behavior_and_conversation.app_data_sources import (
    SpotifyDatabase,
    get_data_generator
)


@dataclass
class AppLogEntry:
    """A single app log entry."""
    timestamp: str  # "YYYY-MM-DD HH:MM:SS"
    app_name: str
    api_name: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


APP_API_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Amazon": {
        "SearchProducts": {
            "input": {"query": "string"},
            "output": {
                "products": [
                    {"name": "string", "price": "number", "rating": "number", "description": "string"}
                ],
                "total_results": "integer"
            }
        },
        "ViewProduct": {
            "input": {"product_name": "string"},
            "output": {
                "name": "string",
                "price": "number",
                "rating": "number",
                "reviews": "integer",
                "description": "string",
                "in_stock": "boolean"
            }
        },
        "PurchaseProduct": {
            "input": {"product_name": "string", "quantity": "integer"},
            "output": {
                "order_number": "string",
                "product_name": "string",
                "quantity": "integer",
                "total_price": "number",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "estimated_delivery": "YYYY-MM-DD"
            }
        },
        "ViewOrders": {
            "input": {},
            "output": {"orders": ["object"], "total_orders": "integer"}
        }
    },
    "Google": {
        "Search": {
            "input": {"query": "string"},
            "output": {
                "results": [{"title": "string", "snippet": "string", "source": "string"}],
                "total_results": "integer"
            }
        }
    },
    "SimpleNote": {
        "ViewNotes": {
            "input": {},
            "output": {
                "notes": [{"title": "string", "preview": "string", "created_at": "string"}],
                "total_notes": "integer"
            }
        },
        "ReadNote": {
            "input": {"note_title": "string"},
            "output": {"title": "string", "content": "string", "created_at": "string"}
        },
        "CreateNote": {
            "input": {"title": "string", "content": "string"},
            "output": {"title": "string", "content": "string", "created_at": "string"}
        }
    },
    "Message": {
        "SendMessage": {
            "input": {"to": "string", "text": "string"},
            "output": {
                "from": "string",
                "to": "string",
                "text": "string",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "status": "string"
            }
        },
        "ViewConversation": {
            "input": {"contact_name": "string"},
            "output": {"contact_name": "string", "messages": ["object"], "total_count": "integer"}
        },
        "SearchMessages": {
            "input": {"query": "string"},
            "output": {"messages": ["object"], "total_count": "integer"}
        }
    },
    "Spotify": {
        "SearchSongs": {
            "input": {"query": "string"},
            "output": {"songs": ["object"], "total_results": "integer"}
        },
        "PlaySong": {
            "input": {"song_title": "string", "artist": "string"},
            "output": {
                "song": "object",
                "status": "string",
                "duration_seconds": "integer"
            }
        },
        "ViewPlaylists": {
            "input": {},
            "output": {"playlists": ["object"]}
        },
        "ViewRecentlyPlayed": {
            "input": {},
            "output": {"songs": ["object"]}
        }
    },
    "Fitness": {
        "LogWorkout": {
            "input": {
                "activity_type": "string",
                "duration_minutes": "integer",
                "intensity": "string"
            },
            "output": {
                "activity_type": "string",
                "duration_minutes": "integer",
                "intensity": "string",
                "calories_burned": "integer",
                "heart_rate_avg": "integer",
                "timestamp": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "ViewTodayStats": {
            "input": {},
            "output": {
                "date": "YYYY-MM-DD",
                "steps": "integer",
                "active_minutes": "integer",
                "calories": "integer",
                "workouts": ["object"]
            }
        },
        "ViewWeeklyStats": {
            "input": {},
            "output": {
                "period": "string",
                "total_workouts": "integer",
                "total_active_minutes": "integer",
                "avg_daily_steps": "integer"
            }
        },
        "SyncDevice": {
            "input": {},
            "output": {
                "steps": "integer",
                "heart_rate_measurements": "integer",
                "sleep_hours": "number",
                "last_sync": "YYYY-MM-DD HH:MM:SS"
            }
        }
    },
    "LLM": {
        "Chat": {
            "input": {
                "message": "string (description: The user's message to the AI assistant)"
            },
            "output": {
                "conversation": [
                    {
                        "role": "string (description: Either 'user' or 'assistant')",
                        "content": "string (description: The message content)"
                    }
                ],
                "description": "string (description: Brief summary of the conversation - what the user asked and what the assistant provided)"
            }
        }
    }
}


class BaseApp(ABC):
    """
    Base class for all apps.

    Each app maintains user-specific state to ensure consistency.
    State is initialized lazily on first API call.
    """

    def __init__(self, app_name: str, user_id: str):
        self.app_name = app_name
        self.user_id = user_id
        self.state: Dict[str, Any] = {}
        self.initialized = False

    @abstractmethod
    def _initialize_state(self) -> None:
        """Initialize app-specific state for this user."""
        pass

    def ensure_initialized(self) -> None:
        """Ensure state is initialized before API calls."""
        if not self.initialized:
            self._initialize_state()
            self.initialized = True

    @abstractmethod
    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """
        Call an API and return a log entry.

        Args:
            api_name: Name of the API to call
            timestamp: When this call happened
            description: LLM-generated description of what user is doing
            context: Additional context (user profile, domain info, etc.)
        """
        pass


class AmazonApp(BaseApp):
    """Amazon shopping app."""

    def __init__(self, user_id: str):
        super().__init__("Amazon", user_id)

    def _initialize_state(self) -> None:
        """Initialize Amazon state."""
        self.state = {
            "order_history": [],
            "search_history": [],
            "viewed_products": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Amazon API."""
        self.ensure_initialized()

        if api_name == "SearchProducts":
            return self._search_products(timestamp, description, context)
        elif api_name == "ViewProduct":
            return self._view_product(timestamp, description, context)
        elif api_name == "PurchaseProduct":
            return self._purchase_product(timestamp, description, context)
        elif api_name == "ViewOrders":
            return self._view_orders(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Amazon API: {api_name}")

    def _search_products(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle product search."""
        query = self._extract_search_query(description)
        products = self._generate_search_results(query, context)

        # Update search history
        self.state["search_history"].append({
            "timestamp": timestamp,
            "query": query,
            "results_count": len(products)
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchProducts",
            request={"query": query},
            response={
                "products": products,
                "total_results": len(products)
            }
        )

    def _view_product(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View product details."""
        product_name = self._extract_product_name(description)
        product = self._get_or_create_product(product_name, context)

        # Add to viewed products
        self.state["viewed_products"].append({
            "timestamp": timestamp,
            "product_name": product["name"]
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewProduct",
            request={"product_name": product_name},
            response=product
        )

    def _purchase_product(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle product purchase."""
        product_name = self._extract_product_name(description)
        quantity = self._extract_quantity(description)
        product = self._get_or_create_product(product_name, context)

        order_number = f"AMZ{random.randint(100000000, 999999999)}"
        total_price = product["price"] * quantity
        order = {
            "order_number": order_number,
            "product_name": product["name"],
            "quantity": quantity,
            "total_price": round(total_price, 2),
            "timestamp": timestamp,
            "estimated_delivery": self._calculate_delivery_date(timestamp)
        }

        # Add to order history
        self.state["order_history"].append(order)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="PurchaseProduct",
            request={"product_name": product_name, "quantity": quantity},
            response=order
        )

    def _view_orders(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View order history."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewOrders",
            request={},
            response={
                "orders": self.state["order_history"][-10:],
                "total_orders": len(self.state["order_history"])
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:5])

    def _extract_product_name(self, description: str) -> str:
        """Extract product name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:5])

    def _extract_quantity(self, description: str) -> int:
        """Extract quantity from description."""
        import re
        numbers = re.findall(r'\d+', description)
        return int(numbers[0]) if numbers else 1

    def _generate_search_results(self, query: str, context: Dict[str, Any]) -> List[Dict]:
        """Generate realistic search results."""
        num_products = random.randint(3, 8)
        products = []

        for i in range(num_products):
            products.append({
                "name": f"{query.title()} - Model {chr(65+i)}",
                "price": round(random.uniform(19.99, 299.99), 2),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "description": f"High-quality {query} product"
            })

        return products

    def _get_or_create_product(self, product_name: str, context: Dict[str, Any]) -> Dict:
        """Get or create product by name."""
        # Check if we have recently viewed this product
        for viewed in reversed(self.state["viewed_products"]):
            if viewed["product_name"].lower() == product_name.lower():
                # Return existing product with same attributes
                return {
                    "name": viewed["product_name"],
                    "price": round(random.uniform(19.99, 299.99), 2),
                    "rating": round(random.uniform(3.5, 5.0), 1),
                    "reviews": random.randint(10, 5000),
                    "description": f"High-quality {viewed['product_name']}",
                    "in_stock": True
                }

        # Create new product
        return {
            "name": product_name,
            "price": round(random.uniform(19.99, 299.99), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews": random.randint(10, 5000),
            "description": f"High-quality {product_name}",
            "in_stock": True
        }

    def _calculate_delivery_date(self, order_timestamp: str) -> str:
        """Calculate delivery date (2-5 days from order)."""
        from datetime import datetime, timedelta
        dt = datetime.strptime(order_timestamp, "%Y-%m-%d %H:%M:%S")
        delivery_dt = dt + timedelta(days=random.randint(2, 5))
        return delivery_dt.strftime("%Y-%m-%d")


class GoogleApp(BaseApp):
    """Google search app."""

    def __init__(self, user_id: str):
        super().__init__("Google", user_id)

    def _initialize_state(self) -> None:
        """Initialize Google state."""
        self.state = {
            "search_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Google API."""
        self.ensure_initialized()

        if api_name == "Search":
            return self._search(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Google API: {api_name}")

    def _search(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle search."""
        query = self._extract_search_query(description)
        results = self._generate_search_results(query)

        self.state["search_history"].append(
            {"timestamp": timestamp, "query": query, "results_count": len(results)}
        )

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Search",
            request={"query": query},
            response={"results": results, "total_results": len(results)}
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:6])

    def _generate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate basic search results without URLs to avoid hallucination."""
        results = []
        sources = ["Wikipedia", "News article", "Academic paper", "Blog post", "Forum discussion", "Official site"]
        for idx in range(random.randint(3, 6)):
            results.append({
                "title": f"{query.title()} - Result {idx + 1}",
                "snippet": f"Relevant information about {query}. This result provides useful context and details.",
                "source": random.choice(sources)
            })
        return results


class SpotifyApp(BaseApp):
    """Spotify music streaming app."""

    def __init__(self, user_id: str):
        super().__init__("Spotify", user_id)

    def _initialize_state(self) -> None:
        """Initialize Spotify state."""
        self.state = {
            "playlists": self._create_initial_playlists(),
            "recently_played": [],
            "favorite_genres": ["Pop", "Rock", "Electronic"]
        }

    def _create_initial_playlists(self) -> List[Dict]:
        """Create some initial playlists."""
        return [
            {
                "name": "My Favorites",
                "song_count": random.randint(20, 100)
            },
            {
                "name": "Workout Mix",
                "song_count": random.randint(15, 50)
            }
        ]

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Spotify API."""
        self.ensure_initialized()

        if api_name == "SearchSongs":
            return self._search_songs(timestamp, description, context)
        elif api_name == "PlaySong":
            return self._play_song(timestamp, description, context)
        elif api_name == "ViewPlaylists":
            return self._view_playlists(timestamp, description, context)
        elif api_name == "ViewRecentlyPlayed":
            return self._view_recently_played(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Spotify API: {api_name}")

    def _search_songs(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for songs."""
        query = self._extract_search_query(description)
        songs = self._generate_song_results(query)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchSongs",
            request={"query": query},
            response={"songs": songs, "total_results": len(songs)}
        )

    def _play_song(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Play a song."""
        song_title, artist = self._extract_song_info(description)
        song = self._get_or_create_song(song_title, artist, context)

        # Add to recently played
        self.state["recently_played"].insert(0, {
            "timestamp": timestamp,
            "song": song
        })

        # Keep only last 50
        self.state["recently_played"] = self.state["recently_played"][:50]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="PlaySong",
            request={"song_title": song_title, "artist": artist},
            response={
                "song": song,
                "status": "playing",
                "duration_seconds": song["duration"]
            }
        )

    def _view_playlists(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View user playlists."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewPlaylists",
            request={},
            response={"playlists": self.state["playlists"]}
        )

    def _view_recently_played(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View recently played songs."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewRecentlyPlayed",
            request={},
            response={
                "songs": [item["song"] for item in self.state["recently_played"][:20]]
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "popular music"

    def _extract_song_info(self, description: str) -> tuple[str, str]:
        """Extract song title and artist from description."""
        import re
        # Try to extract from quotes
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            song_info = quoted[0][0] or quoted[0][1]
            # Try to split by 'by' or '-'
            if " by " in song_info.lower():
                parts = song_info.split(" by ", 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else "Unknown Artist"
            elif " - " in song_info:
                parts = song_info.split(" - ", 1)
                return parts[0].strip(), parts[1].strip() if len(parts) > 1 else "Unknown Artist"
            return song_info, "Unknown Artist"

        # Fallback: use description
        words = description.split()
        return " ".join(words[:3]), "Unknown Artist"

    def _generate_song_results(self, query: str) -> List[Dict]:
        """Generate song search results using static database."""
        songs = SpotifyDatabase.search_songs(query, limit=8)
        return songs

    def _get_or_create_song(self, song_title: str, artist: str, context: Dict[str, Any]) -> Dict:
        """Get or create a song using static database."""
        # Check recently played for exact match
        for item in self.state["recently_played"]:
            song = item["song"]
            if (song.get("title", "").lower() == song_title.lower() and
                song.get("artist", "").lower() == artist.lower()):
                return song

        # Try to find in database
        desc_lower = song_title.lower() + " " + artist.lower()
        if "workout" in desc_lower or "energy" in desc_lower:
            songs = SpotifyDatabase.get_songs_by_genre("Hip Hop", limit=1)
            return songs[0] if songs else SpotifyDatabase.get_random_song()
        elif "chill" in desc_lower or "relax" in desc_lower:
            songs = SpotifyDatabase.get_songs_by_genre("Ambient", limit=1)
            return songs[0] if songs else SpotifyDatabase.get_random_song()
        elif "indie" in desc_lower:
            songs = SpotifyDatabase.get_songs_by_genre("Indie Rock", limit=1)
            return songs[0] if songs else SpotifyDatabase.get_random_song()
        else:
            return SpotifyDatabase.get_random_song()


class SimpleNoteApp(BaseApp):
    """Simple note-taking app."""

    def __init__(self, user_id: str):
        super().__init__("SimpleNote", user_id)

    def _initialize_state(self) -> None:
        """Initialize note state."""
        self.state = {
            "notes": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call SimpleNote API."""
        self.ensure_initialized()

        if api_name == "ViewNotes":
            return self._view_notes(timestamp, description, context)
        elif api_name == "ReadNote":
            return self._read_note(timestamp, description, context)
        elif api_name == "CreateNote":
            return self._create_note(timestamp, description, context)
        else:
            raise ValueError(f"Unknown SimpleNote API: {api_name}")

    def _view_notes(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View all notes."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewNotes",
            request={},
            response={
                "notes": [
                    {
                        "title": note["title"],
                        "preview": note["content"][:50] + ("..." if len(note["content"]) > 50 else ""),
                        "created_at": note["created_at"]
                    }
                    for note in self.state["notes"]
                ],
                "total_notes": len(self.state["notes"])
            }
        )

    def _read_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Read a specific note by title."""
        note_title = self._extract_note_title(description)

        # Find note by title
        note = None
        for n in self.state["notes"]:
            if n["title"].lower() == note_title.lower():
                note = n
                break

        # If not found, use most recent or create placeholder
        if not note:
            if self.state["notes"]:
                note = self.state["notes"][-1]
            else:
                note = {
                    "title": note_title,
                    "content": "This is a placeholder note.",
                    "created_at": timestamp
                }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ReadNote",
            request={"note_title": note_title},
            response=note
        )

    def _create_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a new note."""
        title, content = self._extract_note_content(description)

        note = {
            "title": title,
            "content": content,
            "created_at": timestamp
        }

        self.state["notes"].append(note)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateNote",
            request={"title": title, "content": content},
            response=note
        )

    def _extract_note_title(self, description: str) -> str:
        """Extract note title from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:5])

    def _extract_note_content(self, description: str) -> tuple[str, str]:
        """Extract note title and content from description."""
        words = description.split()
        title = " ".join(words[:5])
        content = description
        return title, content


class MessageApp(BaseApp):
    """Messaging app."""

    def __init__(self, user_id: str):
        super().__init__("Message", user_id)

    def _initialize_state(self) -> None:
        """Initialize messaging state."""
        self.state = {
            "conversations": {},  # contact_name -> list of messages
            "contacts": ["Alice", "Bob", "Carol", "Dave", "Emma"]
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Message API."""
        self.ensure_initialized()

        if api_name == "SendMessage":
            return self._send_message(timestamp, description, context)
        elif api_name == "ViewConversation":
            return self._view_conversation(timestamp, description, context)
        elif api_name == "SearchMessages":
            return self._search_messages(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Message API: {api_name}")

    def _send_message(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Send a message."""
        recipient, message_text = self._extract_message_details(description)

        message = {
            "from": self.user_id,
            "to": recipient,
            "text": message_text,
            "timestamp": timestamp,
            "status": "sent"
        }

        # Add to conversation
        if recipient not in self.state["conversations"]:
            self.state["conversations"][recipient] = []
        self.state["conversations"][recipient].append(message)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SendMessage",
            request={"to": recipient, "text": message_text},
            response=message
        )

    def _view_conversation(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View conversation with a contact."""
        contact_name = self._extract_contact_name(description)
        messages = self.state["conversations"].get(contact_name, [])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewConversation",
            request={"contact_name": contact_name},
            response={
                "contact_name": contact_name,
                "messages": messages[-20:],  # Last 20 messages
                "total_count": len(messages)
            }
        )

    def _search_messages(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search messages."""
        query = self._extract_search_query(description)

        # Simple search through all conversations
        results = []
        for contact, messages in self.state["conversations"].items():
            for msg in messages:
                if query.lower() in msg["text"].lower():
                    results.append(msg)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchMessages",
            request={"query": query},
            response={
                "messages": results[:10],  # Return up to 10 results
                "total_count": len(results)
            }
        )

    def _extract_contact_name(self, description: str) -> str:
        """Extract contact name from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        # Fallback to random contact
        return random.choice(self.state["contacts"]) if self.state["contacts"] else "Friend"

    def _extract_message_details(self, description: str) -> tuple[str, str]:
        """Extract recipient and message from description."""
        import re
        # Try to find quoted message
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        message_text = quoted[0][0] or quoted[0][1] if quoted else description[:100]

        # Extract recipient - look for "to X" pattern
        recipient_match = re.search(r'\bto\s+([A-Z][a-z]+)', description)
        if recipient_match:
            recipient = recipient_match.group(1)
        else:
            recipient = random.choice(self.state["contacts"]) if self.state["contacts"] else "Friend"

        return recipient, message_text

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "search query"


class FitnessApp(BaseApp):
    """Fitness tracking app."""

    def __init__(self, user_id: str):
        super().__init__("Fitness", user_id)

    def _initialize_state(self) -> None:
        """Initialize fitness state."""
        self.state = {
            "workouts": [],
            "daily_stats": {},  # date -> stats
            "goals": {
                "daily_steps": 10000,
                "weekly_workouts": 5,
                "weekly_active_minutes": 150
            },
            "user_baseline": None  # Will be initialized on first use
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Fitness API."""
        self.ensure_initialized()

        if api_name == "LogWorkout":
            return self._log_workout(timestamp, description, context)
        elif api_name == "ViewTodayStats":
            return self._view_today_stats(timestamp, description, context)
        elif api_name == "ViewWeeklyStats":
            return self._view_weekly_stats(timestamp, description, context)
        elif api_name == "SyncDevice":
            return self._sync_device(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Fitness API: {api_name}")

    def _log_workout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Log a workout."""
        # Initialize baseline if needed
        if self.state["user_baseline"] is None:
            self._initialize_user_baseline(context)

        activity_type, duration, intensity = self._extract_workout_details(description)

        workout = {
            "activity_type": activity_type,
            "duration_minutes": duration,
            "intensity": intensity,
            "calories_burned": self._calculate_calories(activity_type, duration, intensity),
            "heart_rate_avg": self._generate_heart_rate(intensity),
            "timestamp": timestamp
        }

        self.state["workouts"].append(workout)

        # Update daily stats
        date = timestamp.split()[0]
        if date not in self.state["daily_stats"]:
            self.state["daily_stats"][date] = {
                "date": date,
                "steps": 0,
                "active_minutes": 0,
                "calories": 0,
                "workouts": []
            }

        self.state["daily_stats"][date]["active_minutes"] += duration
        self.state["daily_stats"][date]["calories"] += workout["calories_burned"]
        self.state["daily_stats"][date]["workouts"].append(workout)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="LogWorkout",
            request={
                "activity_type": activity_type,
                "duration_minutes": duration,
                "intensity": intensity
            },
            response=workout
        )

    def _view_today_stats(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View today's fitness stats."""
        date = timestamp.split()[0]

        if date not in self.state["daily_stats"]:
            # Generate daily stats
            self.state["daily_stats"][date] = {
                "date": date,
                "steps": random.randint(3000, 12000),
                "active_minutes": random.randint(20, 90),
                "calories": random.randint(1800, 2500),
                "workouts": []
            }

        stats = self.state["daily_stats"][date]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewTodayStats",
            request={},
            response=stats
        )

    def _view_weekly_stats(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """View weekly fitness stats."""
        # Aggregate last 7 days
        total_workouts = len(self.state["workouts"][-7:])
        total_active_minutes = sum(w["duration_minutes"] for w in self.state["workouts"][-7:])

        weekly_stats = {
            "period": "last_7_days",
            "total_workouts": total_workouts,
            "total_active_minutes": total_active_minutes,
            "avg_daily_steps": random.randint(6000, 10000)
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ViewWeeklyStats",
            request={},
            response=weekly_stats
        )

    def _sync_device(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Sync fitness device."""
        # Simulate syncing data from wearable
        synced_data = {
            "steps": random.randint(100, 1000),
            "heart_rate_measurements": random.randint(10, 50),
            "sleep_hours": round(random.uniform(6.0, 8.5), 1),
            "last_sync": timestamp
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SyncDevice",
            request={},
            response=synced_data
        )

    def _initialize_user_baseline(self, context: Dict[str, Any]) -> None:
        """Initialize user fitness baseline based on context using LLM data generator."""
        # Use LLM data generator for personalized baseline
        data_gen = get_data_generator()
        user_profile = context.get("user_profile", {})

        # Extract description from context if available
        description = context.get("description", "")

        self.state["user_baseline"] = data_gen.generate_fitness_baseline(
            user_profile=user_profile,
            description=description
        )

    def _extract_workout_details(self, description: str) -> tuple[str, int, str]:
        """Extract workout type, duration, and intensity."""
        # Parse description for workout details
        desc_lower = description.lower()

        # Determine workout type
        if "run" in desc_lower or "jog" in desc_lower:
            workout_type = "running"
        elif "bike" in desc_lower or "cycle" in desc_lower:
            workout_type = "cycling"
        elif "swim" in desc_lower:
            workout_type = "swimming"
        elif "strength" in desc_lower or "weight" in desc_lower:
            workout_type = "strength_training"
        elif "yoga" in desc_lower:
            workout_type = "yoga"
        else:
            workout_type = random.choice(["running", "cycling", "strength_training", "other"])

        # Duration
        duration = random.randint(20, 90)

        # Intensity
        if "intense" in desc_lower or "hard" in desc_lower:
            intensity = "high"
        elif "easy" in desc_lower or "light" in desc_lower:
            intensity = "low"
        else:
            intensity = "moderate"

        return workout_type, duration, intensity

    def _calculate_calories(self, workout_type: str, duration: int, intensity: str) -> int:
        """Calculate calories burned."""
        base_rate = {
            "running": 10,
            "cycling": 8,
            "swimming": 9,
            "strength_training": 6,
            "yoga": 3,
            "other": 5
        }.get(workout_type, 5)

        intensity_multiplier = {"low": 0.7, "moderate": 1.0, "high": 1.3}.get(intensity, 1.0)

        return int(base_rate * duration * intensity_multiplier)

    def _generate_heart_rate(self, intensity: str) -> int:
        """Generate average heart rate for workout."""
        if self.state["user_baseline"]:
            resting_hr = self.state["user_baseline"]["avg_resting_heart_rate"]
        else:
            resting_hr = 65

        intensity_range = {
            "low": (resting_hr + 20, resting_hr + 40),
            "moderate": (resting_hr + 40, resting_hr + 60),
            "high": (resting_hr + 60, resting_hr + 80)
        }.get(intensity, (resting_hr + 30, resting_hr + 50))

        return random.randint(*intensity_range)


class LLMApp(BaseApp):
    """LLM chat app."""

    def __init__(self, user_id: str):
        super().__init__("LLM", user_id)

    def _initialize_state(self) -> None:
        """Initialize LLM state."""
        self.state = {
            "conversation_history": []
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call LLM API."""
        self.ensure_initialized()

        if api_name == "Chat":
            return self._chat(timestamp, description, context)
        else:
            raise ValueError(f"Unknown LLM API: {api_name}")

    def _chat(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle chat - returns full conversation turn."""
        # Extract user message from description
        user_message = self._extract_user_message(description)

        # Generate AI response based on description
        ai_response = self._generate_ai_response(user_message, description, context)

        # Create conversation turn
        conversation_turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ai_response}
        ]

        # Add to conversation history
        self.state["conversation_history"].append({
            "timestamp": timestamp,
            "user_message": user_message,
            "ai_response": ai_response
        })

        # Generate description of the conversation
        conv_description = self._generate_conversation_description(user_message, ai_response)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Chat",
            request={"message": user_message},
            response={
                "conversation": conversation_turn,
                "description": conv_description
            }
        )

    def _extract_user_message(self, description: str) -> str:
        """Extract user message from description."""
        import re
        # Try to extract from quotes
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        # Use description as is
        return description

    def _generate_ai_response(self, user_message: str, description: str, context: Dict[str, Any]) -> str:
        """Generate a plausible AI response using LLM data generator."""
        # Use LLM data generator for chat responses
        data_gen = get_data_generator()
        user_profile = context.get("user_profile", {})

        response = data_gen.generate_llm_chat_response(
            user_message=user_message,
            conversation_history=self.state["conversation_history"],
            user_profile=user_profile
        )

        return response

    def _generate_conversation_description(self, user_message: str, ai_response: str) -> str:
        """Generate a brief description of the conversation."""
        # Extract key topic from user message
        user_msg_short = user_message[:50] + "..." if len(user_message) > 50 else user_message
        ai_resp_short = ai_response[:50] + "..." if len(ai_response) > 50 else ai_response

        return f"User asked about: {user_msg_short}. Assistant provided: {ai_resp_short}"


class AppRegistry:
    """
    Registry for all app instances.

    Maintains one instance per (app_name, user_id) to ensure state consistency
    across all domains and time windows.
    """

    def __init__(self):
        self.apps: Dict[tuple[str, str], BaseApp] = {}

    def get_app(self, app_name: str, user_id: str) -> BaseApp:
        """Get or create an app instance for this user."""
        key = (app_name, user_id)

        if key not in self.apps:
            # Create new app instance
            if app_name == "Amazon":
                self.apps[key] = AmazonApp(user_id)
            elif app_name == "Google":
                self.apps[key] = GoogleApp(user_id)
            elif app_name == "Spotify":
                self.apps[key] = SpotifyApp(user_id)
            elif app_name == "SimpleNote":
                self.apps[key] = SimpleNoteApp(user_id)
            elif app_name == "LLM":
                self.apps[key] = LLMApp(user_id)
            elif app_name == "Message":
                self.apps[key] = MessageApp(user_id)
            elif app_name == "Fitness":
                self.apps[key] = FitnessApp(user_id)
            else:
                raise ValueError(f"Unknown app: {app_name}")

        return self.apps[key]

    def call_api(
        self,
        app_name: str,
        api_name: str,
        user_id: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """
        Call an app API.

        This is the main entry point for generating app logs.
        """
        app = self.get_app(app_name, user_id)
        return app.call_api(api_name, timestamp, description, context)
