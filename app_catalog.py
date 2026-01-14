"""App catalog used by the event chain prompt."""

APP_CATALOG = {
    "apps": [
        {
            "app_name": "Amazon",
            "app_category": "E-commerce",
            "relevant_life_domains": [
                "Finances & Material Living",
                "Leisure & Media Consumption",
            ],
            "apis": [
                {
                    "api_name": "SearchProducts",
                    "description": "Search for products on Amazon using keywords",
                    "user_insights": [
                        "Shopping needs and product interests",
                        "Price sensitivity through search filters",
                        "Product category preferences",
                        "Purchase planning behavior",
                    ],
                },
                {
                    "api_name": "ShowProduct",
                    "description": (
                        "View detailed information about a specific product including price, "
                        "ratings, reviews, and description"
                    ),
                    "user_insights": [
                        "Purchase decision-making process",
                        "Comparison shopping behavior",
                        "Quality consciousness through review reading",
                        "Time spent evaluating options",
                    ],
                },
                {
                    "api_name": "AddToCart",
                    "description": "Add a product to the shopping cart",
                    "user_insights": [
                        "Purchase intent and immediacy",
                        "Impulsive vs. planned buying behavior",
                        "Shopping cart abandonment patterns",
                        "Multi-item purchasing habits",
                    ],
                },
                {
                    "api_name": "ShowCart",
                    "description": "View all items currently in the shopping cart",
                    "user_insights": [
                        "Short-term purchase intentions",
                        "Cart management habits",
                        "Price threshold for checkout",
                        "Multi-session shopping behavior",
                    ],
                    "frequency": "medium",
                },
                {
                    "api_name": "ShowWishlist",
                    "description": (
                        "View all items saved in the wishlist for future consideration"
                    ),
                    "user_insights": [
                        "Long-term purchase aspirations",
                        "Price monitoring behavior",
                        "Gift planning and special occasion preparation",
                        "Delayed gratification patterns",
                    ],
                },
                {
                    "api_name": "Checkout",
                    "description": "Complete the purchase of items in the cart",
                    "user_insights": [
                        "Actual purchasing power and spending",
                        "Buying frequency and volume",
                        "Prime membership utilization",
                        "Payment method preferences",
                    ],
                },
            ],
        },
        {
            "app_name": "Spotify",
            "app_category": "Music Streaming",
            "relevant_life_domains": ["Leisure & Media Consumption"],
            "apis": [
                {
                    "api_name": "SearchSongs",
                    "description": "Search for songs, artists, or albums",
                    "user_insights": [
                        "Music discovery behavior",
                        "Genre preferences and diversity",
                        "Openness to new artists",
                        "Music taste evolution",
                    ],
                },
                {
                    "api_name": "PlaySong",
                    "description": "Play a specific song and track listening duration",
                    "user_insights": [
                        "Core music preferences and listening patterns",
                        "Daily routine and activity timing (workout music, commute, sleep)",
                        "Mood and emotional states",
                        "Song repetition and attachment behavior",
                        "Premium subscription status for ad-free experience",
                    ],
                },
                {
                    "api_name": "AddToPlaylist",
                    "description": "Add a song to a specific playlist",
                    "user_insights": [
                        "Music curation and organization skills",
                        "Long-term music preferences",
                        "Playlist themes and life contexts (workout, study, party)",
                        "Collection-building behavior",
                    ],
                },
                {
                    "api_name": "FollowArtist",
                    "description": (
                        "Follow an artist to receive updates and recommendations"
                    ),
                    "user_insights": [
                        "Artist loyalty and fandom intensity",
                        "Music taste identity and expression",
                        "Social signaling through artist choices",
                        "Engagement with music community",
                    ],
                },
            ],
        },
        {
            "app_name": "Fitbit",
            "app_category": "Health & Fitness Tracking",
            "relevant_life_domains": ["Health & Self-care"],
            "apis": [
                {
                    "api_name": "LogWorkout",
                    "description": (
                        "Manually log a workout session with type, duration, and intensity"
                    ),
                    "user_insights": [
                        "Active fitness engagement and discipline",
                        "Preferred exercise types and variety",
                        "Workout intensity preferences",
                        "Self-tracking motivation and consistency",
                    ],
                },
                {
                    "api_name": "SyncDevice",
                    "description": (
                        "Sync wearable device data including steps, heart rate, sleep patterns, "
                        "and passive activity"
                    ),
                    "user_insights": [
                        "Daily activity levels and sedentary behavior",
                        "Sleep quality and schedule regularity",
                        "Cardiovascular health awareness",
                        "Technology adoption for health monitoring",
                    ],
                },
                {
                    "api_name": "SetGoals",
                    "description": (
                        "Set or update fitness goals such as daily steps, active minutes, "
                        "or weight targets"
                    ),
                    "user_insights": [
                        "Health ambitions and self-expectations",
                        "Goal-setting realism vs. optimism",
                        "Commitment to lifestyle changes",
                        "Self-improvement priorities",
                    ],
                },
            ],
        },
        {
            "app_name": "Chase",
            "app_category": "Banking & Financial Management",
            "relevant_life_domains": ["Finances & Material Living"],
            "apis": [
                {
                    "api_name": "GetBalance",
                    "description": "Check current account balance",
                    "user_insights": [
                        "Financial awareness and monitoring frequency",
                        "Money anxiety or security levels",
                        "Account checking habits as stress indicator",
                        "Financial buffer comfort zone",
                    ],
                    "frequency": "high",
                },
                {
                    "api_name": "GetTransactions",
                    "description": (
                        "View recent transaction history with merchant details, amounts, "
                        "and categories"
                    ),
                    "user_insights": [
                        "Spending patterns across categories (dining, shopping, transportation)",
                        "Financial responsibility and tracking behavior",
                        "Lifestyle spending priorities",
                        "Cash flow management awareness",
                    ],
                    "frequency": "medium",
                },
                {
                    "api_name": "SearchTransactions",
                    "description": (
                        "Search for specific transactions by merchant, amount, or date range"
                    ),
                    "user_insights": [
                        "Active financial management and recordkeeping",
                        "Expense dispute or verification needs",
                        "Tax preparation or budgeting diligence",
                        "Financial organization skills",
                    ],
                    "frequency": "low",
                },
                {
                    "api_name": "TransferMoney",
                    "description": "Transfer funds between accounts or to other people",
                    "user_insights": [
                        "Liquidity management strategies",
                        "Savings discipline and allocation",
                        "Financial support relationships (family, friends)",
                        "Multi-account optimization behavior",
                    ],
                },
                {
                    "api_name": "PayBill",
                    "description": "Pay bills such as utilities, credit cards, or subscriptions",
                    "user_insights": [
                        "Financial responsibility and payment timeliness",
                        "Recurring expense patterns",
                        "Bill management automation preferences",
                        "Essential vs. discretionary spending balance",
                    ],
                },
            ],
        },
        {
            "app_name": "Robinhood",
            "app_category": "Investment & Trading",
            "relevant_life_domains": ["Finances & Material Living"],
            "apis": [
                {
                    "api_name": "GetPortfolio",
                    "description": (
                        "View current investment holdings, positions, and portfolio value"
                    ),
                    "user_insights": [
                        "Investment style (aggressive vs. conservative)",
                        "Asset diversification sophistication",
                        "Portfolio monitoring frequency and anxiety",
                        "Wealth accumulation and investment commitment",
                    ],
                },
                {
                    "api_name": "GetWatchlist",
                    "description": (
                        "View list of stocks or crypto being monitored for potential investment"
                    ),
                    "user_insights": [
                        "Investment research and planning behavior",
                        "Market sector interests",
                        "Risk appetite indicators through watchlist choices",
                        "Patient vs. impulsive investing approach",
                    ],
                },
                {
                    "api_name": "SearchStocks",
                    "description": (
                        "Search for stocks or crypto by symbol or company name"
                    ),
                    "user_insights": [
                        "Active investment research intensity",
                        "Market opportunity exploration",
                        "Financial curiosity and learning engagement",
                        "New investment consideration frequency",
                    ],
                },
                {
                    "api_name": "GetStockQuote",
                    "description": (
                        "View current price, change, and details for a specific stock or crypto"
                    ),
                    "user_insights": [
                        "Market monitoring habits and timing",
                        "Price sensitivity and entry point strategy",
                        "Information-seeking before decisions",
                        "Investment due diligence thoroughness",
                    ],
                },
                {
                    "api_name": "BuyStock",
                    "description": "Execute a purchase of stocks or crypto",
                    "user_insights": [
                        "Investment decision-making confidence",
                        "Capital deployment aggressiveness",
                        "Market timing beliefs and behavior",
                        "Financial risk tolerance in action",
                    ],
                },
                {
                    "api_name": "SellStock",
                    "description": "Execute a sale of stocks or crypto",
                    "user_insights": [
                        "Profit-taking vs. loss-cutting discipline",
                        "Emotional response to market volatility",
                        "Exit strategy sophistication",
                        "Portfolio rebalancing awareness",
                    ],
                },
            ],
        },
        {
            "app_name": "WhatsApp",
            "app_category": "Instant Messaging",
            "relevant_life_domains": [
                "Family & Close Relationships",
                "Social & Community",
            ],
            "apis": [
                {
                    "api_name": "GetMessages",
                    "description": (
                        "Retrieve message history from a specific contact or group"
                    ),
                    "user_insights": [
                        "Communication frequency with different relationships",
                        "Relationship intimacy and depth through message volume",
                        "Conversation review and reminiscence behavior",
                        "Social network structure and priority contacts",
                    ],
                },
                {
                    "api_name": "SendMessage",
                    "description": "Send a text message to a contact or group",
                    "user_insights": [
                        "Communication initiation patterns",
                        "Message length and conversation depth preferences",
                        "Response speed and availability signals",
                        "Relationship maintenance effort and priorities",
                    ],
                },
                {
                    "api_name": "SendMedia",
                    "description": "Send photos, videos, or voice messages",
                    "user_insights": [
                        "Rich communication preferences",
                        "Life moment sharing behavior",
                        "Visual vs. text communication style",
                        "Intimacy expression through media types",
                    ],
                },
            ],
        },
        {
            "app_name": "Gmail",
            "app_category": "Email",
            "relevant_life_domains": ["Work & Education", "Social & Community"],
            "apis": [
                {
                    "api_name": "GetInbox",
                    "description": (
                        "Retrieve current inbox emails with previews and metadata"
                    ),
                    "user_insights": [
                        "Email volume as work intensity indicator",
                        "Inbox management style (inbox zero vs. accumulator)",
                        "Information overload levels",
                        "Professional communication burden",
                    ],
                    "frequency": "very_high",
                },
                {
                    "api_name": "ReadEmail",
                    "description": "Open and read a specific email",
                    "user_insights": [
                        "Email prioritization and triage decisions",
                        "Information processing speed",
                        "Attention allocation to different senders",
                        "Email response time patterns",
                    ],
                },
                {
                    "api_name": "SendEmail",
                    "description": "Compose and send a new email",
                    "user_insights": [
                        "Proactive communication and initiative",
                        "Professional relationship building",
                        "Email formality and communication style",
                        "Work productivity and output generation",
                    ],
                },
                {
                    "api_name": "ReplyEmail",
                    "description": "Reply to a received email",
                    "user_insights": [
                        "Responsiveness and reliability",
                        "Communication reciprocity patterns",
                        "Reply speed by sender relationship",
                        "Professional courtesy and engagement",
                    ],
                },
            ],
        },
        {
            "app_name": "LinkedIn",
            "app_category": "Professional Networking",
            "relevant_life_domains": ["Work & Education", "Social & Community"],
            "apis": [
                {
                    "api_name": "UpdateProfile",
                    "description": (
                        "Update profile information such as headline, summary, or photo"
                    ),
                    "user_insights": [
                        "Personal branding awareness and effort",
                        "Career positioning and messaging",
                        "Professional identity evolution",
                        "Job market readiness signals",
                    ],
                },
                {
                    "api_name": "AddExperience",
                    "description": "Add or update work experience entries",
                    "user_insights": [
                        "Career progression and mobility",
                        "Achievement documentation habits",
                        "Professional milestone celebration",
                        "Resume maintenance discipline",
                    ],
                },
                {
                    "api_name": "AddSkill",
                    "description": "Add new skills to profile",
                    "user_insights": [
                        "Skill development and learning focus areas",
                        "Career development strategy",
                        "Professional growth mindset",
                        "Market positioning through skill signals",
                    ],
                },
                {
                    "api_name": "PostUpdate",
                    "description": "Share a post, article, or thought on LinkedIn feed",
                    "user_insights": [
                        "Thought leadership aspirations",
                        "Professional content creation and sharing",
                        "Industry engagement and visibility efforts",
                        "Personal brand building activity",
                    ],
                },
                {
                    "api_name": "GetFeed",
                    "description": (
                        "View LinkedIn feed with posts from connections and followed pages"
                    ),
                    "user_insights": [
                        "Professional content consumption habits",
                        "Industry news and trend awareness",
                        "Learning and development engagement",
                        "Professional network monitoring",
                    ],
                },
                {
                    "api_name": "LikePost",
                    "description": "Like a post in the feed",
                    "user_insights": [
                        "Content preference signals",
                        "Network engagement and support behavior",
                        "Professional relationship nurturing",
                        "Visibility and presence maintenance",
                    ],
                },
                {
                    "api_name": "CommentOnPost",
                    "description": (
                        "Comment on a post to share thoughts or engage in discussion"
                    ),
                    "user_insights": [
                        "Deep engagement with professional content",
                        "Thought leadership and expertise demonstration",
                        "Network relationship deepening efforts",
                        "Discussion participation willingness",
                    ],
                },
                {
                    "api_name": "SearchJobs",
                    "description": (
                        "Search for job openings by keywords, location, or company"
                    ),
                    "user_insights": [
                        "Active job seeking status and intensity",
                        "Career change considerations",
                        "Job market exploration and dissatisfaction signals",
                        "Career goals and aspirations",
                    ],
                },
                {
                    "api_name": "ApplyJob",
                    "description": "Submit application for a job posting",
                    "user_insights": [
                        "Serious job transition intent",
                        "Job application volume and selectivity",
                        "Career change readiness",
                        "Job search commitment level",
                    ],
                },
                {
                    "api_name": "SendConnectionRequest",
                    "description": (
                        "Send a connection request to another LinkedIn user"
                    ),
                    "user_insights": [
                        "Networking proactivity and strategy",
                        "Professional relationship building efforts",
                        "Career network expansion goals",
                        "Social capital investment behavior",
                    ],
                },
            ],
        },
        {
            "app_name": "Notion",
            "app_category": "Knowledge Management & Productivity",
            "relevant_life_domains": ["Work & Education", "Health & Self-care"],
            "apis": [
                {
                    "api_name": "GetPages",
                    "description": "Retrieve list of pages and notebooks in workspace",
                    "user_insights": [
                        "Knowledge management system scope",
                        "Organization complexity and structure",
                        "Content creation volume and diversity",
                        "Digital workspace organization style",
                    ],
                },
                {
                    "api_name": "CreatePage",
                    "description": "Create a new page or note",
                    "user_insights": [
                        "Knowledge production and documentation habits",
                        "Note-taking frequency and triggers",
                        "Thinking and learning process externalization",
                        "Creative or analytical work patterns",
                    ],
                },
                {
                    "api_name": "UpdatePage",
                    "description": "Edit and update existing page content",
                    "user_insights": [
                        "Iterative thinking and refinement behavior",
                        "Content maintenance and quality standards",
                        "Knowledge evolution and updates tracking",
                        "Perfectionism vs. completion tendencies",
                    ],
                },
                {
                    "api_name": "SearchContent",
                    "description": "Search across all pages and databases",
                    "user_insights": [
                        "Information retrieval efficiency needs",
                        "Knowledge reuse and reference behavior",
                        "Memory reliance vs. search dependence",
                        "Information organization effectiveness",
                    ],
                },
                {
                    "api_name": "CreateDatabaseEntry",
                    "description": (
                        "Add entry to a database (task, project, habit tracker, etc.)"
                    ),
                    "user_insights": [
                        "Structured productivity and tracking systems",
                        "Task and project management discipline",
                        "Quantified self and habit tracking behavior",
                        "Goal-oriented planning and execution",
                    ],
                },
            ],
        },
        {
            "app_name": "Netflix",
            "app_category": "Video Streaming",
            "relevant_life_domains": ["Leisure & Media Consumption"],
            "apis": [
                {
                    "api_name": "SearchContent",
                    "description": "Search for movies, TV shows, or documentaries",
                    "user_insights": [
                        "Active content discovery preferences",
                        "Genre and topic interests",
                        "Specific viewing intent vs. browsing",
                        "Decision-making approach for entertainment",
                    ],
                    "frequency": "medium",
                },
                {
                    "api_name": "ShowTitle",
                    "description": (
                        "View detailed information about a specific title including description, "
                        "cast, and ratings"
                    ),
                    "user_insights": [
                        "Content evaluation thoroughness",
                        "Decision-making deliberation for viewing",
                        "Quality consciousness and selectivity",
                        "Time spent on content selection",
                    ],
                },
                {
                    "api_name": "PlayContent",
                    "description": "Start playing a movie or TV show episode",
                    "user_insights": [
                        "Viewing frequency and binge-watching patterns",
                        "Content preferences and genre tastes",
                        "Viewing time distribution (weekday vs. weekend, time of day)",
                        "Watch duration and completion rates",
                        "Subscription tier for streaming quality",
                    ],
                },
                {
                    "api_name": "AddToMyList",
                    "description": "Add a title to personal watchlist",
                    "user_insights": [
                        "Content curation and planning behavior",
                        "Delayed viewing intentions",
                        "Aspiration vs. actual viewing gap",
                        "List management and follow-through",
                    ],
                },
                {
                    "api_name": "RateContent",
                    "description": "Rate a watched title with thumbs up or down",
                    "user_insights": [
                        "Feedback and opinion expression willingness",
                        "Algorithm training engagement",
                        "Content evaluation standards and taste clarity",
                        "Platform interaction and investment",
                    ],
                },
            ],
        },
        {
            "app_name": "Goodreads",
            "app_category": "Book Tracking & Reviews",
            "relevant_life_domains": [
                "Leisure & Media Consumption",
                "Work & Education",
            ],
            "apis": [
                {
                    "api_name": "SearchBooks",
                    "description": "Search for books by title, author, or keywords",
                    "user_insights": [
                        "Reading interests and topic preferences",
                        "Book discovery methods (recommendations vs. direct search)",
                        "Genre preferences and reading diversity",
                        "Intellectual curiosity areas",
                    ],
                },
                {
                    "api_name": "ShowBook",
                    "description": (
                        "View detailed information about a specific book including synopsis, "
                        "ratings, and reviews"
                    ),
                    "user_insights": [
                        "Reading decision-making thoroughness",
                        "Book selection criteria and standards",
                        "Review reliance and opinion-seeking",
                        "Quality consciousness for reading material",
                    ],
                },
                {
                    "api_name": "AddToShelf",
                    "description": (
                        "Add a book to a specific shelf (want-to-read, currently-reading, read)"
                    ),
                    "user_insights": [
                        "Reading planning and intention setting",
                        "Book collection curation behavior",
                        "Reading progress tracking discipline",
                        "Aspirational vs. actual reading habits",
                    ],
                },
                {
                    "api_name": "RateBook",
                    "description": "Rate a book on a 1-5 star scale",
                    "user_insights": [
                        "Reading engagement and completion",
                        "Critical thinking and evaluation skills",
                        "Rating standards and generosity",
                        "Personal taste clarity and confidence",
                    ],
                    "frequency": "low",
                },
                {
                    "api_name": "WriteReview",
                    "description": "Write a text review for a book",
                    "user_insights": [
                        "Deep reflection on reading experience",
                        "Written expression and articulation skills",
                        "Willingness to share opinions publicly",
                        "Intellectual engagement depth with material",
                    ],
                    "frequency": "very_low",
                },
            ],
        },
        {
            "app_name": "Instagram",
            "app_category": "Social Media & Photo Sharing",
            "relevant_life_domains": [
                "Social & Community",
                "Leisure & Media Consumption",
            ],
            "apis": [
                {
                    "api_name": "PostStory",
                    "description": (
                        "Post a photo or video to Instagram Stories (24-hour temporary content)"
                    ),
                    "user_insights": [
                        "Daily life sharing frequency and openness",
                        "Casual vs. curated content preferences",
                        "Social presence maintenance",
                        "Ephemeral vs. permanent sharing comfort",
                    ],
                },
                {
                    "api_name": "LikePost",
                    "description": "Like a post in the feed",
                    "user_insights": [
                        "Social engagement levels and generosity",
                        "Content consumption patterns and interests",
                        "Relationship acknowledgment behavior",
                        "Feed scrolling depth and time",
                    ],
                },
                {
                    "api_name": "CommentOnPost",
                    "description": "Comment on a post",
                    "user_insights": [
                        "Deep social engagement willingness",
                        "Relationship investment and maintenance",
                        "Public communication comfort",
                        "Thoughtfulness in interactions",
                    ],
                },
                {
                    "api_name": "SendDirectMessage",
                    "description": (
                        "Send a private message to another user"
                    ),
                    "user_insights": [
                        "Private communication preferences",
                        "Content sharing behavior (memes, posts, personal messages)",
                        "Close friendship maintenance",
                        "Social initiation patterns",
                    ],
                },
                {
                    "api_name": "FollowUser",
                    "description": "Follow another user's account",
                    "user_insights": [
                        "Social network expansion behavior",
                        "Interest-based following vs. social obligation",
                        "Content curation preferences",
                        "New relationship openness",
                    ],
                },
                {
                    "api_name": "UnfollowUser",
                    "description": "Unfollow a user's account",
                    "user_insights": [
                        "Social network curation and pruning",
                        "Relationship ending or distancing",
                        "Content quality standards enforcement",
                        "Digital boundary setting",
                    ],
                },
                {
                    "api_name": "GetFollowing",
                    "description": "View list of accounts currently followed",
                    "user_insights": [
                        "Social network size and composition review",
                        "Following audit and cleanup consideration",
                        "Social comparison behavior",
                        "Network management awareness",
                    ],
                },
            ],
        },
        {
            "app_name": "Google",
            "app_category": "Search Engine",
            "life_domains": ["All domains - cross-cutting tool"],
            "apis": [
                {
                    "api_name": "Search",
                    "description": "Perform a web search and receive list of results",
                    "user_insights": [
                        "Information needs and curiosity areas",
                        "Search query formulation sophistication",
                        "Problem-solving approach (search vs. ask AI)",
                        "Fact-checking and verification habits",
                    ],
                },
                {
                    "api_name": "ClickResult",
                    "description": (
                        "Click on a specific search result to view the webpage"
                    ),
                    "user_insights": [
                        "Result evaluation and selection criteria",
                        "Source trustworthiness judgment",
                        "Information gathering depth",
                        "Click position bias (top results vs. deeper exploration)",
                    ],
                },
            ],
        },
        {
            "app_name": "LLM Assistant",
            "app_category": "AI Assistant",
            "relevant_life_domains": ["All domains - cross-cutting tool"],
            "apis": [
                {
                    "api_name": "CreateConversation",
                    "description": "Start a new conversation thread with the AI assistant",
                    "user_insights": [
                        "Task switching and compartmentalization",
                        "New problem or topic initiation",
                        "AI usage frequency and dependency",
                        "Conversation organization preferences",
                    ],
                },
                {
                    "api_name": "ContinueConversation",
                    "description": (
                        "Send a message in an existing conversation thread"
                    ),
                    "user_insights": [
                        "Conversation continuity and depth",
                        "Complex task breakdown and iteration",
                        "Clarification and refinement patterns",
                        "Multi-turn interaction engagement",
                        "Query types and domains (work, learning, creative, personal)",
                        "Problem-solving approach and follow-through",
                    ],
                },
            ],
        },
    ]
}
