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
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string",
                "membership": "string"
            }
        },
        "SearchProducts": {
            "input": {
                "query": "string",
                "session_id": "string"
            },
            "output": {
                "products": [
                    {
                        "product_id": "string",
                        "name": "string",
                        "price": "number",
                        "rating": "number",
                    }
                ],
                "results_count": "integer"
            }
        },
        "ShowProduct": {
            "input": {
                "product_id": "string",
                "session_id": "string"
            },
            "output": {
                "product_id": "string",
                "name": "string",
                "price": "number",
                "rating": "number",
                "reviews_count": "integer",
                "description": "string"
            }
        },
        "AddToCart": {
            "input": {
                "product_id": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "cart": "object"
            }
        },
        "Checkout": {
            "input": {
                "product_id": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "order_id": "string",
                "product": "object",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "status": "string",
                "delivery_date": "YYYY-MM-DD"
            }
        },
        "ShowOrders": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "orders": ["object"],
                "orders_count": "integer",
                "total_orders": "integer"
            }
        }
    },
    "Walmart": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string",
                "membership": "string"
            }
        },
        "SearchProducts": {
            "input": {
                "query": "string",
                "session_id": "string"
            },
            "output": {
                "products": [
                    {
                        "product_id": "string",
                        "name": "string",
                        "price": "number",
                        "rating": "number",
                    }
                ],
                "results_count": "integer"
            }
        },
        "ShowProduct": {
            "input": {
                "product_id": "string",
                "session_id": "string"
            },
            "output": {
                "product_id": "string",
                "name": "string",
                "price": "number",
                "rating": "number",
                "reviews_count": "integer",
                "description": "string"
            }
        },
        "AddToCart": {
            "input": {
                "product_id": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "cart": "object"
            }
        },
        "Checkout": {
            "input": {
                "product_id": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "order_id": "string",
                "product": "object",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "status": "string",
                "delivery_date": "YYYY-MM-DD"
            }
        },
        "ShowOrders": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "orders": ["object"],
                "orders_count": "integer",
                "total_orders": "integer"
            }
        }
    },
    "Google": {
        "Search": {
            "input": {
                "query": "string"
            },
            "output": {
                "results": [
                    {
                        "title": "string",
                        "url": "string",
                        "snippet": "string"
                    }
                ],
                "results_count": "integer"
            }
        }
    },
    "Gmail": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string"
            }
        },
        "GetNewMails": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "emails": [
                    {
                        "email_id": "string",
                        "sender": "string",
                        "subject": "string",
                        "snippet": "string",
                        "timestamp": "YYYY-MM-DD HH:MM:SS"
                    }
                ],
                "emails_count": "integer"
            }
        },
        "FetchEmails": {
            "input": {
                "query": "string",
                "limit": "integer",
                "session_id": "string"
            },
            "output": {
                "emails": [
                    {
                        "email_id": "string",
                        "sender": "string",
                        "subject": "string",
                        "snippet": "string",
                        "timestamp": "YYYY-MM-DD HH:MM:SS"
                    }
                ]
            }
        },
        "SendEmail": {
            "input": {
                "to": "string",
                "subject": "string",
                "body": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "message_id": "string"
            }
        }
    },
    "SimpleNote": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string"
            }
        },
        "ShowAllNotes": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "notes": [
                    {
                        "note_id": "string",
                        "title": "string",
                        "preview": "string",
                        "created_at": "YYYY-MM-DD HH:MM:SS"
                    }
                ],
                "notes_count": "integer"
            }
        },
        "ShowNote": {
            "input": {
                "note_id": "string",
                "session_id": "string"
            },
            "output": {
                "note_id": "string",
                "title": "string",
                "content": "string",
                "created_at": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "CreateNote": {
            "input": {
                "title": "string",
                "content": "string",
                "session_id": "string"
            },
            "output": {
                "note_id": "string",
                "title": "string",
                "content": "string",
                "created_at": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "UpdateNote": {
            "input": {
                "note_id": "string",
                "title": "string",
                "content": "string",
                "session_id": "string"
            },
            "output": {
                "note_id": "string",
                "title": "string",
                "content": "string",
                "updated_at": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "DeleteNote": {
            "input": {
                "note_id": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "deleted_note_id": "string"
            }
        }
    },
    "Google Calendar": {
        "CreateEvent": {
            "input": {
                "title": "string",
                "start_time": "YYYY-MM-DD HH:MM:SS",
                "duration_minutes": "integer",
                "session_id": "string"
            },
            "output": {
                "event_id": "string",
                "title": "string",
                "start_time": "YYYY-MM-DD HH:MM:SS",
                "duration_minutes": "integer",
                "status": "string",
                "created_at": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "UpdateEvent": {
            "input": {
                "event_id": "string",
                "session_id": "string"
            },
            "output": {
                "event_id": "string",
                "title": "string",
                "updated_at": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "DeleteEvent": {
            "input": {
                "event_id": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "deleted_event_id": "string"
            }
        },
        "ShowAllEvents": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "events": ["object"],
                "total_count": "integer"
            }
        },
        "ShowEvent": {
            "input": {
                "event_id": "string",
                "session_id": "string"
            },
            "output": {
                "event_id": "string",
                "title": "string",
                "start_time": "YYYY-MM-DD HH:MM:SS",
                "duration_minutes": "integer",
                "status": "string"
            }
        },
        "RespondToInvite": {
            "input": {
                "invite_id": "string",
                "response": "accepted|declined",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "status": "string"
            }
        }
    },
    "WhatsApp": {
        "ReadNewMessages": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "messages": ["object"],
                "messages_count": "integer"
            }
        },
        "SendMessage": {
            "input": {
                "to": "string",
                "text": "string",
                "session_id": "string"
            },
            "output": {
                "message_id": "string",
                "from": "string",
                "to": "string",
                "text": "string",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "status": "string"
            }
        },
        "SearchMessages": {
            "input": {
                "query": "string",
                "session_id": "string"
            },
            "output": {
                "messages": ["object"],
                "total_count": "integer"
            }
        },
        "GetMessages": {
            "input": {
                "to": "string",
                "session_id": "string"
            },
            "output": {
                "to": "string",
                "messages": ["object"],
                "messages_count": "integer"
            }
        },
        "CreateGroup": {
            "input": {
                "name": "string",
                "members": ["string"],
                "session_id": "string"
            },
            "output": {
                "group_id": "string",
                "name": "string",
                "members": ["string"],
                "created_at": "YYYY-MM-DD HH:MM:SS",
                "created_by": "string"
            }
        }
    },
    "ChatGPT": {
        "GetConversations": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "conversations": ["object"],
                "conversations_count": "integer"
            }
        },
        "NewChat": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "conversation_id": "string"
            }
        },
        "LoadChat": {
            "input": {
                "conversation_id": "string",
                "session_id": "string"
            },
            "output": {
                "conversation_id": "string",
                "messages": ["object"],
                "messages_count": "integer"
            }
        },
        "SendMessage": {
            "input": {
                "conversation_id": "string",
                "message": "string",
                "session_id": "string"
            },
            "output": {
                "message_id": "string",
                "conversation_id": "string",
                "message": "string",
                "Response": "string"
            }
        }
    },
    "Spotify": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string",
                "premium": "boolean"
            }
        },
        "SearchSongs": {
            "input": {
                "query": "string",
                "session_id": "string"
            },
            "output": {
                "songs": [
                    {
                        "song_id": "string",
                        "title": "string",
                        "artist": "string",
                        "album": "string",
                        "duration": "integer"
                    }
                ],
                "total_results": "integer"
            }
        },
        "PlaySong": {
            "input": {
                "song_id": "string",
                "session_id": "string"
            },
            "output": {
                "song": "object",
                "status": "string",
                "duration_seconds": "integer"
            }
        },
        "ShowPlaylists": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "playlists": [
                    {
                        "playlist_id": "string",
                        "name": "string",
                        "song_count": "integer"
                    }
                ]
            }
        },
        "ShowRecentlyPlayed": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "songs": ["object"]
            }
        }
    },
    "Fitbit": {
        "LogWorkout": {
            "input": {
                "type": "string",
                "duration_minutes": "integer",
                "intensity": "string",
                "session_id": "string"
            },
            "output": {
                "workout_id": "string",
                "type": "string",
                "duration_minutes": "integer",
                "intensity": "string",
                "calories_burned": "integer",
                "heart_rate_avg": "integer",
                "timestamp": "YYYY-MM-DD HH:MM:SS",
                "date": "YYYY-MM-DD"
            }
        },
        "GetDailyStats": {
            "input": {
                "date": "YYYY-MM-DD",
                "session_id": "string"
            },
            "output": {
                "date": "YYYY-MM-DD",
                "steps": "integer",
                "active_minutes": "integer",
                "calories": "integer",
                "workouts": ["string"]
            }
        },
        "GetWeeklyStats": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "period": "string",
                "total_workouts": "integer",
                "total_active_minutes": "integer",
                "avg_daily_steps": "integer",
                "goal_completion": "object"
            }
        },
        "GetWorkoutHistory": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "workouts": ["object"],
                "total_count": "integer"
            }
        },
        "UpdateGoals": {
            "input": {
                "goals": "object",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "goals": "object"
            }
        },
        "SyncDevice": {
            "input": {
                "device_name": "string",
                "session_id": "string"
            },
            "output": {
                "steps": "integer",
                "heart_rate_measurements": "integer",
                "sleep_hours": "number",
                "last_sync": "YYYY-MM-DD HH:MM:SS"
            }
        }
    },
    "Chase": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string"
            }
        },
        "GetTransactions": {
            "input": {
                "session_id": "string",
                "month": "string"
            },
            "output": {
                "transactions": [
                    {
                        "transaction_id": "string",
                        "date": "YYYY-MM-DD",
                        "description": "string",
                        "amount": "number",
                        "type": "debit|credit"
                    }
                ],
                "balance": "number"
            }
        },
        "TransferMoney": {
            "input": {
                "to_account": "string",
                "amount": "number",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "transaction_id": "string",
                "new_balance": "number"
            }
        },
        "PayBill": {
            "input": {
                "biller_name": "string",
                "amount": "number",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "transaction_id": "string"
            }
        }
    },
    "Robinhood": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string",
                "portfolio_value": "number"
            }
        },
        "GetPortfolio": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "cash_balance": "number",
                "holdings": [
                    {
                        "symbol": "string",
                        "quantity": "integer",
                        "current_price": "number",
                        "total_value": "number"
                    }
                ],
                "total_value": "number"
            }
        },
        "GetStockQuote": {
            "input": {
                "symbol": "string",
                "session_id": "string"
            },
            "output": {
                "symbol": "string",
                "price": "number",
                "change_percent": "number",
                "timestamp": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "BuyStock": {
            "input": {
                "symbol": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "transaction_id": "string",
                "price": "number",
                "total_cost": "number"
            }
        },
        "SellStock": {
            "input": {
                "symbol": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "transaction_id": "string",
                "price": "number",
                "total_proceeds": "number"
            }
        },
        "BuyCrypto": {
            "input": {
                "symbol": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "transaction_id": "string",
                "price": "number",
                "total_cost": "number"
            }
        },
        "SellCrypto": {
            "input": {
                "symbol": "string",
                "quantity": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "transaction_id": "string",
                "price": "number",
                "total_proceeds": "number"
            }
        },
        "GetHistory": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "transactions": ["object"],
                "count": "integer"
            }
        }
    },
    "X": {
        "Login": {
            "input": {
                "username": "string",
                "password": "string"
            },
            "output": {
                "success": "boolean",
                "session_id": "string",
                "user_id": "string"
            }
        },
        "GetHomeTimeline": {
            "input": {
                "session_id": "string"
            },
            "output": {
                "posts": [
                    {
                        "post_id": "string",
                        "author": "string",
                        "content": "string",
                        "likes": "integer",
                        "timestamp": "YYYY-MM-DD HH:MM:SS"
                    }
                ]
            }
        },
        "PostTweet": {
            "input": {
                "content": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "post_id": "string",
                "timestamp": "YYYY-MM-DD HH:MM:SS"
            }
        },
        "LikeTweet": {
            "input": {
                "post_id": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "likes_count": "integer"
            }
        },
        "FollowUser": {
            "input": {
                "user_id": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "follow_status": "string"
            }
        },
        "GetUserProfile": {
            "input": {
                "user_id": "string",
                "session_id": "string"
            },
            "output": {
                "username": "string",
                "followers": "integer",
                "following": "integer",
                "bio": "string"
            }
        }
    },
    "Google Home": {
        "SetLightState": {
            "input": {
                "device_id": "string",
                "state": "on|off",
                "brightness": "integer",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "device_id": "string",
                "state": "string"
            }
        },
        "GetDeviceStatus": {
            "input": {
                "device_id": "string",
                "session_id": "string"
            },
            "output": {
                "device_id": "string",
                "name": "string",
                "type": "string",
                "state": "string",
                "is_online": "boolean"
            }
        },
        "RunRoutine": {
            "input": {
                "routine_name": "string",
                "session_id": "string"
            },
            "output": {
                "success": "boolean",
                "routine_name": "string",
                "actions_executed": ["string"]
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
            "session_id": f"amz_session_{uuid.uuid4().hex[:16]}",
            "order_history": [],
            "search_history": [],
            "cart": [],
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

        if api_name == "Login":
            return self._login(timestamp, description, context)
        elif api_name == "SearchProducts":
            return self._search_products(timestamp, description, context)
        elif api_name == "ShowProduct":
            return self._show_product(timestamp, description, context)
        elif api_name == "Checkout":
            return self._checkout(timestamp, description, context)
        elif api_name == "ShowOrders":
            return self._show_orders(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Amazon API: {api_name}")

    def _login(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle login."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Login",
            request={"user_id": self.user_id},
            response={
                "success": True,
                "session_id": self.state["session_id"],
                "user_name": context.get("user_name", f"User_{self.user_id[:8]}")
            }
        )

    def _search_products(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle product search."""
        # Extract search query from description
        query = self._extract_search_query(description)

        # Generate realistic search results
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
            request={"query": query, "session_id": self.state["session_id"]},
            response={
                "products": products,
                "total_results": len(products)
            }
        )

    def _show_product(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show product details."""
        product = self._extract_or_generate_product(description, context)

        # Add to viewed products
        self.state["viewed_products"].append({
            "timestamp": timestamp,
            "product_id": product["product_id"],
            "product_name": product["name"]
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowProduct",
            request={
                "product_id": product["product_id"],
                "session_id": self.state["session_id"]
            },
            response=product
        )

    def _checkout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle checkout."""
        product = self._extract_or_generate_product(description, context)

        order_id = f"AMZ{random.randint(100000000, 999999999)}"
        order = {
            "order_id": order_id,
            "product": product,
            "timestamp": timestamp,
            "status": "confirmed",
            "delivery_date": self._calculate_delivery_date(timestamp)
        }

        # Add to order history
        self.state["order_history"].append(order)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Checkout",
            request={
                "product_id": product["product_id"],
                "quantity": 1,
                "session_id": self.state["session_id"]
            },
            response=order
        )

    def _show_orders(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show order history."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowOrders",
            request={"session_id": self.state["session_id"]},
            response={
                "orders": self.state["order_history"][-10:],  # Last 10 orders
                "total_orders": len(self.state["order_history"])
            }
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        # Simple extraction - can be improved
        desc_lower = description.lower()
        if "search" in desc_lower:
            # Try to find quoted text
            import re
            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
            if quoted:
                return quoted[0][0] or quoted[0][1]

        # Fallback: use key words from description
        words = description.split()
        return " ".join(words[:5])

    def _generate_search_results(self, query: str, context: Dict[str, Any]) -> List[Dict]:
        """Generate realistic search results."""
        # Generate 3-8 products
        num_products = random.randint(3, 8)
        products = []

        for i in range(num_products):
            product_id = f"B{random.randint(10000000, 99999999)}"
            products.append({
                "product_id": product_id,
                "name": f"{query.title()} - Model {chr(65+i)}",
                "price": round(random.uniform(19.99, 299.99), 2),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "reviews": random.randint(10, 5000)
            })

        return products

    def _extract_or_generate_product(self, description: str, context: Dict[str, Any]) -> Dict:
        """Extract product from description or generate one."""
        # Check if we have recently viewed products
        if self.state["viewed_products"]:
            # Reuse a recent product
            recent = self.state["viewed_products"][-1]
            product_id = recent["product_id"]
            product_name = recent["product_name"]
        else:
            # Generate new product
            product_id = f"B{random.randint(10000000, 99999999)}"
            product_name = description.split()[:5]
            product_name = " ".join(product_name)

        return {
            "product_id": product_id,
            "name": product_name,
            "price": round(random.uniform(19.99, 299.99), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews": random.randint(10, 5000),
            "description": f"High-quality {product_name}"
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
            "session_id": f"google_session_{uuid.uuid4().hex[:16]}",
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
            request={"query": query, "session_id": self.state["session_id"]},
            response={"results": results, "total_results": len(results)}
        )

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        desc_lower = description.lower()
        if "search" in desc_lower or "research" in desc_lower:
            import re
            quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
            if quoted:
                return quoted[0][0] or quoted[0][1]
        words = description.split()
        return " ".join(words[:6])

    def _generate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate basic search results."""
        results = []
        for idx in range(random.randint(3, 6)):
            results.append(
                {
                    "title": f"{query} result {idx + 1}",
                    "url": f"https://example.com/{query.replace(' ', '_').lower()}/{idx + 1}",
                    "snippet": f"Summary about {query} (result {idx + 1})."
                }
            )
        return results


class SpotifyApp(BaseApp):
    """Spotify music streaming app."""

    def __init__(self, user_id: str):
        super().__init__("Spotify", user_id)

    def _initialize_state(self) -> None:
        """Initialize Spotify state."""
        self.state = {
            "session_id": f"spotify_session_{uuid.uuid4().hex[:16]}",
            "playlists": self._create_initial_playlists(),
            "recently_played": [],
            "favorite_genres": ["Pop", "Rock", "Electronic"]
        }

    def _create_initial_playlists(self) -> List[Dict]:
        """Create some initial playlists."""
        return [
            {
                "playlist_id": f"pl_{uuid.uuid4().hex[:16]}",
                "name": "My Favorites",
                "song_count": random.randint(20, 100)
            },
            {
                "playlist_id": f"pl_{uuid.uuid4().hex[:16]}",
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

        if api_name == "Login":
            return self._login(timestamp, description, context)
        elif api_name == "SearchSongs":
            return self._search_songs(timestamp, description, context)
        elif api_name == "PlaySong":
            return self._play_song(timestamp, description, context)
        elif api_name == "ShowPlaylists":
            return self._show_playlists(timestamp, description, context)
        elif api_name == "ShowRecentlyPlayed":
            return self._show_recently_played(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Spotify API: {api_name}")

    def _login(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle login."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Login",
            request={"user_id": self.user_id},
            response={
                "success": True,
                "session_id": self.state["session_id"],
                "premium": True
            }
        )

    def _search_songs(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Search for songs."""
        query = self._extract_search_query(description)
        songs = self._generate_song_results(query)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="SearchSongs",
            request={"query": query, "session_id": self.state["session_id"]},
            response={"songs": songs, "total_results": len(songs)}
        )

    def _play_song(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Play a song."""
        song = self._extract_or_generate_song(description, context)

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
            request={
                "song_id": song["song_id"],
                "session_id": self.state["session_id"]
            },
            response={
                "song": song,
                "status": "playing",
                "duration_seconds": song["duration"]
            }
        )

    def _show_playlists(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show user playlists."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowPlaylists",
            request={"session_id": self.state["session_id"]},
            response={"playlists": self.state["playlists"]}
        )

    def _show_recently_played(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show recently played songs."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowRecentlyPlayed",
            request={"session_id": self.state["session_id"]},
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

    def _generate_song_results(self, query: str) -> List[Dict]:
        """Generate song search results using static database."""
        # Use Spotify static database
        songs = SpotifyDatabase.search_songs(query, limit=8)
        return songs

    def _extract_or_generate_song(self, description: str, context: Dict[str, Any]) -> Dict:
        """Extract or generate a song using static database."""
        # Check recently played
        if self.state["recently_played"] and random.random() < 0.3:
            return self.state["recently_played"][0]["song"]

        # Extract genre/mood from description if possible
        desc_lower = description.lower()
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
            # Return random song from database
            return SpotifyDatabase.get_random_song()


class SimpleNoteApp(BaseApp):
    """Simple note-taking app."""

    def __init__(self, user_id: str):
        super().__init__("SimpleNote", user_id)

    def _initialize_state(self) -> None:
        """Initialize note state."""
        self.state = {
            "session_id": f"note_session_{uuid.uuid4().hex[:16]}",
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

        if api_name == "Login":
            return self._login(timestamp, description, context)
        elif api_name == "ShowNotes":
            return self._show_notes(timestamp, description, context)
        elif api_name == "ShowNote":
            return self._show_note(timestamp, description, context)
        elif api_name == "CreateNote":
            return self._create_note(timestamp, description, context)
        else:
            raise ValueError(f"Unknown SimpleNote API: {api_name}")

    def _login(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Handle login."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Login",
            request={"user_id": self.user_id},
            response={
                "success": True,
                "session_id": self.state["session_id"]
            }
        )

    def _show_notes(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show all notes."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowNotes",
            request={"session_id": self.state["session_id"]},
            response={
                "notes": [
                    {
                        "note_id": note["note_id"],
                        "title": note["title"],
                        "preview": note["content"][:50] + "...",
                        "created_at": note["created_at"]
                    }
                    for note in self.state["notes"]
                ],
                "total_notes": len(self.state["notes"])
            }
        )

    def _show_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show a specific note."""
        # Get the most recent note or generate placeholder
        if self.state["notes"]:
            note = self.state["notes"][-1]
        else:
            note = {
                "note_id": f"note_{uuid.uuid4().hex[:16]}",
                "title": "Placeholder Note",
                "content": "This is a placeholder note.",
                "created_at": timestamp
            }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowNote",
            request={
                "note_id": note["note_id"],
                "session_id": self.state["session_id"]
            },
            response=note
        )

    def _create_note(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a new note."""
        # Extract title and content from description
        title, content = self._extract_note_content(description)

        note = {
            "note_id": f"note_{uuid.uuid4().hex[:16]}",
            "title": title,
            "content": content,
            "created_at": timestamp
        }

        self.state["notes"].append(note)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateNote",
            request={
                "title": title,
                "content": content,
                "session_id": self.state["session_id"]
            },
            response=note
        )

    def _extract_note_content(self, description: str) -> tuple[str, str]:
        """Extract note title and content from description."""
        # Simple extraction
        words = description.split()
        title = " ".join(words[:5])
        content = description
        return title, content


class CalendarApp(BaseApp):
    """Calendar app for event management."""

    def __init__(self, user_id: str):
        super().__init__("Calendar", user_id)

    def _initialize_state(self) -> None:
        """Initialize calendar state."""
        self.state = {
            "session_id": f"cal_session_{uuid.uuid4().hex[:16]}",
            "events": [],
            "next_event_id": 1
        }

    def call_api(
        self,
        api_name: str,
        timestamp: str,
        description: str,
        context: Dict[str, Any]
    ) -> AppLogEntry:
        """Call Calendar API."""
        self.ensure_initialized()

        if api_name == "CreateEvent":
            return self._create_event(timestamp, description, context)
        elif api_name == "UpdateEvent":
            return self._update_event(timestamp, description, context)
        elif api_name == "DeleteEvent":
            return self._delete_event(timestamp, description, context)
        elif api_name == "ShowEvents":
            return self._show_events(timestamp, description, context)
        elif api_name == "ShowEvent":
            return self._show_event(timestamp, description, context)
        elif api_name == "RespondToInvite":
            return self._respond_to_invite(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Calendar API: {api_name}")

    def _create_event(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a new calendar event."""
        event_id = f"evt_{self.state['next_event_id']}"
        self.state['next_event_id'] += 1

        # Extract event details from description
        event_title, event_time, event_duration = self._extract_event_details(description, timestamp)

        new_event = {
            "event_id": event_id,
            "title": event_title,
            "start_time": event_time,
            "duration_minutes": event_duration,
            "status": "confirmed",
            "created_at": timestamp
        }

        self.state["events"].append(new_event)

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateEvent",
            request={
                "title": event_title,
                "start_time": event_time,
                "duration_minutes": event_duration,
                "session_id": self.state["session_id"]
            },
            response=new_event
        )

    def _update_event(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Update an existing event."""
        if self.state["events"]:
            event = self.state["events"][-1]  # Update most recent event
            event["updated_at"] = timestamp
        else:
            event = {"event_id": "evt_0", "title": "Placeholder", "updated_at": timestamp}

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="UpdateEvent",
            request={"event_id": event["event_id"], "session_id": self.state["session_id"]},
            response=event
        )

    def _delete_event(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Delete an event."""
        deleted_id = None
        if self.state["events"]:
            event = self.state["events"].pop()
            deleted_id = event["event_id"]

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="DeleteEvent",
            request={"event_id": deleted_id or "evt_0", "session_id": self.state["session_id"]},
            response={"success": True, "deleted_event_id": deleted_id}
        )

    def _show_events(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show all events."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowEvents",
            request={"session_id": self.state["session_id"]},
            response={
                "events": self.state["events"],
                "total_count": len(self.state["events"])
            }
        )

    def _show_event(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Show a specific event."""
        event = self.state["events"][-1] if self.state["events"] else {
            "event_id": "evt_0",
            "title": "Placeholder Event"
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="ShowEvent",
            request={"event_id": event["event_id"], "session_id": self.state["session_id"]},
            response=event
        )

    def _respond_to_invite(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Respond to an event invite."""
        response_status = "accepted" if "accept" in description.lower() else "declined"

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="RespondToInvite",
            request={
                "invite_id": f"inv_{random.randint(1000, 9999)}",
                "response": response_status,
                "session_id": self.state["session_id"]
            },
            response={"success": True, "status": response_status}
        )

    def _extract_event_details(self, description: str, default_time: str) -> tuple[str, str, int]:
        """Extract event title, time, and duration from description."""
        # Simple extraction
        words = description.split()
        title = " ".join(words[:5]) if len(words) >= 5 else description

        # Use default time, add 1 day for future event
        from datetime import datetime, timedelta
        dt = datetime.strptime(default_time, "%Y-%m-%d %H:%M:%S")
        event_dt = dt + timedelta(days=1)
        event_time = event_dt.strftime("%Y-%m-%d %H:%M:%S")

        duration = random.choice([30, 60, 90, 120])  # minutes

        return title, event_time, duration


class MessageApp(BaseApp):
    """Messaging app."""

    def __init__(self, user_id: str):
        super().__init__("Message", user_id)

    def _initialize_state(self) -> None:
        """Initialize messaging state."""
        self.state = {
            "session_id": f"msg_session_{uuid.uuid4().hex[:16]}",
            "conversations": {},  # contact_id -> list of messages
            "groups": {},
            "contacts": ["friend_1", "friend_2", "family_1", "colleague_1"]
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
        elif api_name == "SearchMessages":
            return self._search_messages(timestamp, description, context)
        elif api_name == "GetMessages":
            return self._get_messages(timestamp, description, context)
        elif api_name == "CreateGroup":
            return self._create_group(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Message API: {api_name}")

    def _send_message(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Send a message."""
        # Extract recipient and message from description
        recipient, message_text = self._extract_message_details(description)

        message = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
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
            request={
                "to": recipient,
                "text": message_text,
                "session_id": self.state["session_id"]
            },
            response=message
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
            request={"query": query, "session_id": self.state["session_id"]},
            response={
                "messages": results[:10],  # Return up to 10 results
                "total_count": len(results)
            }
        )

    def _get_messages(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Get messages from a conversation."""
        # Get random contact or use first contact
        contact = random.choice(self.state["contacts"]) if self.state["contacts"] else "friend_1"
        messages = self.state["conversations"].get(contact, [])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetMessages",
            request={"contact_id": contact, "session_id": self.state["session_id"]},
            response={
                "contact_id": contact,
                "messages": messages[-20:],  # Last 20 messages
                "total_count": len(messages)
            }
        )

    def _create_group(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Create a group chat."""
        group_id = f"grp_{uuid.uuid4().hex[:12]}"
        group_name = self._extract_group_name(description)

        members = random.sample(self.state["contacts"], min(3, len(self.state["contacts"])))
        members.append(self.user_id)

        group = {
            "group_id": group_id,
            "name": group_name,
            "members": members,
            "created_at": timestamp,
            "created_by": self.user_id
        }

        self.state["groups"][group_id] = group

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="CreateGroup",
            request={
                "name": group_name,
                "members": members,
                "session_id": self.state["session_id"]
            },
            response=group
        )

    def _extract_message_details(self, description: str) -> tuple[str, str]:
        """Extract recipient and message from description."""
        # Simple extraction
        recipient = random.choice(self.state["contacts"]) if self.state["contacts"] else "friend_1"
        message_text = description[:100]  # Use description as message
        return recipient, message_text

    def _extract_search_query(self, description: str) -> str:
        """Extract search query from description."""
        import re
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", description)
        if quoted:
            return quoted[0][0] or quoted[0][1]
        return "search query"

    def _extract_group_name(self, description: str) -> str:
        """Extract group name from description."""
        words = description.split()
        return " ".join(words[:3]) if len(words) >= 3 else "New Group"


class FitnessApp(BaseApp):
    """Fitness tracking app."""

    def __init__(self, user_id: str):
        super().__init__("Fitness", user_id)

    def _initialize_state(self) -> None:
        """Initialize fitness state."""
        self.state = {
            "session_id": f"fit_session_{uuid.uuid4().hex[:16]}",
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
        elif api_name == "GetDailyStats":
            return self._get_daily_stats(timestamp, description, context)
        elif api_name == "GetWeeklyStats":
            return self._get_weekly_stats(timestamp, description, context)
        elif api_name == "GetWorkoutHistory":
            return self._get_workout_history(timestamp, description, context)
        elif api_name == "UpdateGoals":
            return self._update_goals(timestamp, description, context)
        elif api_name == "SyncDevice":
            return self._sync_device(timestamp, description, context)
        else:
            raise ValueError(f"Unknown Fitness API: {api_name}")

    def _log_workout(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Log a workout."""
        # Initialize baseline if needed
        if self.state["user_baseline"] is None:
            self._initialize_user_baseline(context)

        workout_type, duration, intensity = self._extract_workout_details(description)

        workout = {
            "workout_id": f"wkt_{uuid.uuid4().hex[:12]}",
            "type": workout_type,
            "duration_minutes": duration,
            "intensity": intensity,
            "calories_burned": self._calculate_calories(workout_type, duration, intensity),
            "heart_rate_avg": self._generate_heart_rate(intensity),
            "timestamp": timestamp,
            "date": timestamp.split()[0]
        }

        self.state["workouts"].append(workout)

        # Update daily stats
        date = workout["date"]
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
        self.state["daily_stats"][date]["workouts"].append(workout["workout_id"])

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="LogWorkout",
            request={
                "type": workout_type,
                "duration_minutes": duration,
                "intensity": intensity,
                "session_id": self.state["session_id"]
            },
            response=workout
        )

    def _get_daily_stats(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Get daily fitness stats."""
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
            api_name="GetDailyStats",
            request={"date": date, "session_id": self.state["session_id"]},
            response=stats
        )

    def _get_weekly_stats(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Get weekly fitness stats."""
        # Aggregate last 7 days
        total_workouts = len(self.state["workouts"][-7:])
        total_active_minutes = sum(w["duration_minutes"] for w in self.state["workouts"][-7:])

        weekly_stats = {
            "period": "last_7_days",
            "total_workouts": total_workouts,
            "total_active_minutes": total_active_minutes,
            "avg_daily_steps": random.randint(6000, 10000),
            "goal_completion": {
                "workouts": f"{total_workouts}/{self.state['goals']['weekly_workouts']}",
                "active_minutes": f"{total_active_minutes}/{self.state['goals']['weekly_active_minutes']}"
            }
        }

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetWeeklyStats",
            request={"session_id": self.state["session_id"]},
            response=weekly_stats
        )

    def _get_workout_history(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Get workout history."""
        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="GetWorkoutHistory",
            request={"session_id": self.state["session_id"]},
            response={
                "workouts": self.state["workouts"][-20:],  # Last 20 workouts
                "total_count": len(self.state["workouts"])
            }
        )

    def _update_goals(self, timestamp: str, description: str, context: Dict[str, Any]) -> AppLogEntry:
        """Update fitness goals."""
        # Extract new goals from description
        new_goals = self.state["goals"].copy()

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="UpdateGoals",
            request={"goals": new_goals, "session_id": self.state["session_id"]},
            response={"success": True, "goals": new_goals}
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
            request={"device_id": "device_001", "session_id": self.state["session_id"]},
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
            "session_id": f"llm_session_{uuid.uuid4().hex[:16]}",
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
        """Handle chat."""
        # Extract user message from description
        user_message = self._extract_user_message(description)

        # Generate a plausible AI response based on description
        ai_response = self._generate_ai_response(user_message, description, context)

        # Add to conversation history
        self.state["conversation_history"].append({
            "timestamp": timestamp,
            "user_message": user_message,
            "ai_response": ai_response
        })

        return AppLogEntry(
            timestamp=timestamp,
            app_name=self.app_name,
            api_name="Chat",
            request={
                "message": user_message,
                "session_id": self.state["session_id"]
            },
            response={
                "reply": ai_response,
                "conversation_id": self.state["session_id"]
            }
        )

    def _extract_user_message(self, description: str) -> str:
        """Extract user message from description."""
        # The description typically describes what the user asks
        if "asks" in description.lower() or "discusses" in description.lower():
            # Try to extract the question/topic
            return description
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
            elif app_name == "Calendar":
                self.apps[key] = CalendarApp(user_id)
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
