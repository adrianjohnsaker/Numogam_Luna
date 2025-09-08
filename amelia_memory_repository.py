import json
import datetime
from typing import List, Dict, Optional, Any, Tuple
import os
from pathlib import Path
import re
import gzip
import shutil
from collections import Counter
import hashlib
    """
    Memory module for storing and retrieving conversation transcripts
    between user and Amelia AI.
    """
    
    def __init__(self, storage_path: str = "amelia_memory"):
        """
        Initialize the memory module.
        
        Args:
            storage_path: Directory path for storing memory files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.conversations_file = self.storage_path / "conversations.json"
        self.memory_index_file = self.storage_path / "memory_index.json"
        self.archive_path = self.storage_path / "archives"
        self.archive_path.mkdir(exist_ok=True)
        self.current_session_id = None
        self._initialize_storage()
        
        # Pruning settings
        self.max_active_conversations = 100
        self.max_message_age_days = 365
        self.max_storage_mb = 500
    
    def _initialize_storage(self):
        """Initialize storage files if they don't exist."""
        if not self.conversations_file.exists():
            self._save_json(self.conversations_file, {})
        
        if not self.memory_index_file.exists():
            self._save_json(self.memory_index_file, {
                "total_conversations": 0,
                "topics": {},
                "keywords": {},
                "last_updated": None
            })
    
    def _save_json(self, filepath: Path, data: Dict):
        """Save data to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def start_conversation(self, user_id: str = "default") -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            session_id: Unique identifier for this conversation
        """
        timestamp = datetime.datetime.now().isoformat()
        session_id = f"{user_id}_{timestamp}"
        self.current_session_id = session_id
        
        conversations = self._load_json(self.conversations_file)
        conversations[session_id] = {
            "user_id": user_id,
            "started_at": timestamp,
            "ended_at": None,
            "messages": [],
            "summary": None,
            "topics": []
        }
        self._save_json(self.conversations_file, conversations)
        
        return session_id
    
    def add_message(self, role: str, content: str, session_id: Optional[str] = None) -> bool:
        """
        Add a message to the current or specified conversation.
        
        Args:
            role: Either "user" or "amelia"
            content: The message content
            session_id: Optional session ID (uses current if not provided)
            
        Returns:
            Success status
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            return False
        
        conversations = self._load_json(self.conversations_file)
        
        if session_id not in conversations:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        conversations[session_id]["messages"].append(message)
        self._save_json(self.conversations_file, conversations)
        
        # Update index with keywords
        self._update_keywords(session_id, content)
        
        return True
    
    def _update_keywords(self, session_id: str, content: str):
        """Extract and index keywords from message content."""
        # Simple keyword extraction (can be enhanced with NLP)
        words = content.lower().split()
        keywords = [w for w in words if len(w) > 4]  # Simple filter
        
        index = self._load_json(self.memory_index_file)
        
        for keyword in keywords:
            if keyword not in index["keywords"]:
                index["keywords"][keyword] = []
            if session_id not in index["keywords"][keyword]:
                index["keywords"][keyword].append(session_id)
        
        index["last_updated"] = datetime.datetime.now().isoformat()
        self._save_json(self.memory_index_file, index)
    
    def end_conversation(self, summary: Optional[str] = None, topics: Optional[List[str]] = None):
        """
        End the current conversation session.
        
        Args:
            summary: Optional summary of the conversation
            topics: Optional list of topics discussed
        """
        if self.current_session_id is None:
            return
        
        conversations = self._load_json(self.conversations_file)
        
        if self.current_session_id in conversations:
            conversations[self.current_session_id]["ended_at"] = datetime.datetime.now().isoformat()
            
            if summary:
                conversations[self.current_session_id]["summary"] = summary
            
            if topics:
                conversations[self.current_session_id]["topics"] = topics
                # Update topic index
                self._update_topic_index(self.current_session_id, topics)
            
            self._save_json(self.conversations_file, conversations)
        
        self.current_session_id = None
    
    def _update_topic_index(self, session_id: str, topics: List[str]):
        """Update the topic index."""
        index = self._load_json(self.memory_index_file)
        
        for topic in topics:
            if topic not in index["topics"]:
                index["topics"][topic] = []
            if session_id not in index["topics"][topic]:
                index["topics"][topic].append(session_id)
        
        index["total_conversations"] = len(self._load_json(self.conversations_file))
        self._save_json(self.memory_index_file, index)
    
    def search_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search conversations containing a specific keyword.
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            List of matching conversation excerpts
        """
        keyword = keyword.lower()
        index = self._load_json(self.memory_index_file)
        
        if keyword not in index["keywords"]:
            return []
        
        conversations = self._load_json(self.conversations_file)
        results = []
        
        for session_id in index["keywords"][keyword]:
            if session_id in conversations:
                conv = conversations[session_id]
                matching_messages = [
                    msg for msg in conv["messages"] 
                    if keyword in msg["content"].lower()
                ]
                
                results.append({
                    "session_id": session_id,
                    "started_at": conv["started_at"],
                    "summary": conv.get("summary"),
                    "matching_messages": matching_messages[:3]  # Limit to 3 messages
                })
        
        return results
    
    def search_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """
        Search conversations by topic.
        
        Args:
            topic: The topic to search for
            
        Returns:
            List of matching conversations
        """
        index = self._load_json(self.memory_index_file)
        
        if topic not in index["topics"]:
            return []
        
        conversations = self._load_json(self.conversations_file)
        results = []
        
        for session_id in index["topics"][topic]:
            if session_id in conversations:
                conv = conversations[session_id]
                results.append({
                    "session_id": session_id,
                    "started_at": conv["started_at"],
                    "summary": conv.get("summary"),
                    "topics": conv.get("topics", []),
                    "message_count": len(conv["messages"])
                })
        
        return results
    
    def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a full conversation by session ID.
        
        Args:
            session_id: The conversation session ID
            
        Returns:
            The conversation data or None if not found
        """
        conversations = self._load_json(self.conversations_file)
        return conversations.get(session_id)
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent conversations.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of recent conversations
        """
        conversations = self._load_json(self.conversations_file)
        
        # Sort by start time
        sorted_convs = sorted(
            conversations.items(),
            key=lambda x: x[1]["started_at"],
            reverse=True
        )
        
        results = []
        for session_id, conv in sorted_convs[:limit]:
            results.append({
                "session_id": session_id,
                "started_at": conv["started_at"],
                "ended_at": conv.get("ended_at"),
                "summary": conv.get("summary"),
                "message_count": len(conv["messages"]),
                "topics": conv.get("topics", [])
            })
        
        return results
    
    def get_conversation_context(self, session_id: Optional[str] = None, 
                                message_limit: int = 10) -> List[Dict[str, str]]:
        """
        Get recent conversation context for Amelia to reference.
        
        Args:
            session_id: Optional session ID (uses current if not provided)
            message_limit: Maximum number of recent messages to return
            
        Returns:
            List of recent messages
        """
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id is None:
            return []
        
        conversation = self.get_conversation(session_id)
        if conversation is None:
            return []
        
        return conversation["messages"][-message_limit:]
    
    def export_memory(self, export_path: str) -> bool:
        """
        Export all memory data to a specified location.
        
        Args:
            export_path: Path to export the memory data
            
        Returns:
            Success status
        """
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(exist_ok=True)
            
            # Copy conversation and index files
            import shutil
            shutil.copy2(self.conversations_file, export_dir / "conversations.json")
            shutil.copy2(self.memory_index_file, export_dir / "memory_index.json")
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def import_conversation_text(self, text: str, user_id: str = "default", 
                               timestamp: Optional[str] = None,
                               format_type: str = "auto") -> str:
        """
        Import a past conversation from text format.
        
        Args:
            text: The conversation text
            user_id: User identifier
            timestamp: Optional timestamp for the conversation
            format_type: Format type - "auto", "simple", "tagged", or "timestamped"
            
        Returns:
            session_id of the imported conversation
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        session_id = f"imported_{user_id}_{timestamp}"
        
        # Parse the conversation based on format
        messages = self._parse_conversation_text(text, format_type)
        
        # Create conversation entry
        conversations = self._load_json(self.conversations_file)
        conversations[session_id] = {
            "user_id": user_id,
            "started_at": timestamp,
            "ended_at": timestamp,
            "messages": messages,
            "summary": None,
            "topics": [],
            "imported": True
        }
        
        self._save_json(self.conversations_file, conversations)
        
        # Update indices
        for msg in messages:
            self._update_keywords(session_id, msg["content"])
        
        return session_id
    
    def _parse_conversation_text(self, text: str, format_type: str) -> List[Dict[str, str]]:
        """Parse conversation text into message format."""
        messages = []
        
        if format_type == "auto":
            # Try to detect format
            if re.search(r'\[?\d{4}-\d{2}-\d{2}', text):
                format_type = "timestamped"
            elif re.search(r'(User:|Amelia:)', text, re.IGNORECASE):
                format_type = "tagged"
            else:
                format_type = "simple"
        
        if format_type == "simple":
            # Alternate between user and amelia
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                role = "user" if i % 2 == 0 else "amelia"
                messages.append({
                    "role": role,
                    "content": line,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        elif format_type == "tagged":
            # Look for User: and Amelia: tags
            pattern = r'(User|Amelia):\s*(.*?)(?=(?:User|Amelia):|$)'
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            
            for role_name, content in matches:
                role = "user" if role_name.lower() == "user" else "amelia"
                messages.append({
                    "role": role,
                    "content": content.strip(),
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        elif format_type == "timestamped":
            # Parse timestamped format [2024-01-15 14:30:00] User: Message
            pattern = r'\[?(\d{4}-\d{2}-\d{2}[\s\dT:.-]*)\]?\s*(User|Amelia):\s*(.*?)(?=\[?\d{4}-\d{2}-\d{2}|$)'
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            
            for timestamp_str, role_name, content in matches:
                role = "user" if role_name.lower() == "user" else "amelia"
                try:
                    # Try to parse timestamp
                    timestamp = datetime.datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                    timestamp_iso = timestamp.isoformat()
                except:
                    timestamp_iso = datetime.datetime.now().isoformat()
                
                messages.append({
                    "role": role,
                    "content": content.strip(),
                    "timestamp": timestamp_iso
                })
        
        return messages
    
    def bulk_import_conversations(self, file_path: str, format_type: str = "auto") -> Dict[str, Any]:
        """
        Bulk import multiple conversations from a file.
        
        Args:
            file_path: Path to file containing conversations
            format_type: Format type for parsing
            
        Returns:
            Import statistics
        """
        import_stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "session_ids": []
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to split into individual conversations
        # Look for conversation separators
        separators = [
            '\n---\n',
            '\n###\n', 
            '\n\n\n',
            'END_CONVERSATION'
        ]
        
        conversations = [content]  # Default to single conversation
        
        for sep in separators:
            if sep in content:
                conversations = content.split(sep)
                break
        
        for i, conv_text in enumerate(conversations):
            if conv_text.strip():
                import_stats["total_attempted"] += 1
                try:
                    session_id = self.import_conversation_text(
                        conv_text.strip(),
                        user_id="imported",
                        timestamp=f"import_{i}_{datetime.datetime.now().isoformat()}",
                        format_type=format_type
                    )
                    import_stats["successful"] += 1
                    import_stats["session_ids"].append(session_id)
                except Exception as e:
                    import_stats["failed"] += 1
                    print(f"Failed to import conversation {i}: {e}")
        
        return import_stats
    
    def prune_old_conversations(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Prune old conversations based on age and storage limits.
        
        Args:
            dry_run: If True, only report what would be pruned without actually removing
            
        Returns:
            Pruning statistics
        """
        conversations = self._load_json(self.conversations_file)
        current_time = datetime.datetime.now()
        
        prune_stats = {
            "total_conversations": len(conversations),
            "conversations_to_prune": 0,
            "messages_to_prune": 0,
            "storage_saved_mb": 0,
            "archived_conversations": []
        }
        
        # Identify conversations to prune
        conversations_by_age = []
        
        for session_id, conv in conversations.items():
            if conv.get("ended_at"):
                try:
                    ended_time = datetime.datetime.fromisoformat(conv["ended_at"])
                    age_days = (current_time - ended_time).days
                    
                    conversations_by_age.append({
                        "session_id": session_id,
                        "age_days": age_days,
                        "message_count": len(conv["messages"]),
                        "has_summary": conv.get("summary") is not None
                    })
                except:
                    continue
        
        # Sort by age (oldest first)
        conversations_by_age.sort(key=lambda x: x["age_days"], reverse=True)
        
        # Determine which conversations to prune
        to_prune = []
        
        # Prune by age
        for conv_info in conversations_by_age:
            if conv_info["age_days"] > self.max_message_age_days:
                to_prune.append(conv_info["session_id"])
        
        # Prune by count if still over limit
        if len(conversations) - len(to_prune) > self.max_active_conversations:
            remaining_count = len(conversations) - len(to_prune) - self.max_active_conversations
            for conv_info in conversations_by_age:
                if conv_info["session_id"] not in to_prune and remaining_count > 0:
                    to_prune.append(conv_info["session_id"])
                    remaining_count -= 1
        
        prune_stats["conversations_to_prune"] = len(to_prune)
        
        if not dry_run and to_prune:
            # Archive conversations before pruning
            archived = self._archive_conversations(to_prune, conversations)
            prune_stats["archived_conversations"] = archived
            
            # Remove from active storage
            for session_id in to_prune:
                del conversations[session_id]
            
            self._save_json(self.conversations_file, conversations)
            
            # Update indices
            self._rebuild_indices()
        
        # Calculate storage saved
        for session_id in to_prune:
            if session_id in conversations:
                prune_stats["messages_to_prune"] += len(conversations[session_id]["messages"])
        
        # Rough estimate: 1KB per message
        prune_stats["storage_saved_mb"] = prune_stats["messages_to_prune"] * 0.001
        
        return prune_stats
    
    def _archive_conversations(self, session_ids: List[str], conversations: Dict) -> List[str]:
        """Archive conversations to compressed storage."""
        archived = []
        
        for session_id in session_ids:
            if session_id in conversations:
                # Create archive entry with summary
                archive_entry = {
                    "session_id": session_id,
                    "user_id": conversations[session_id].get("user_id"),
                    "started_at": conversations[session_id].get("started_at"),
                    "ended_at": conversations[session_id].get("ended_at"),
                    "summary": conversations[session_id].get("summary"),
                    "topics": conversations[session_id].get("topics", []),
                    "message_count": len(conversations[session_id]["messages"]),
                    "sample_messages": conversations[session_id]["messages"][:5]  # Keep first 5 messages
                }
                
                # Save full conversation to compressed file
                archive_file = self.archive_path / f"{session_id}.json.gz"
                with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
                    json.dump(conversations[session_id], f)
                
                archived.append(session_id)
                
                # Update archive index
                archive_index_file = self.archive_path / "archive_index.json"
                if archive_index_file.exists():
                    with open(archive_index_file, 'r') as f:
                        archive_index = json.load(f)
                else:
                    archive_index = {}
                
                archive_index[session_id] = archive_entry
                
                with open(archive_index_file, 'w') as f:
                    json.dump(archive_index, f, indent=2)
        
        return archived
    
    def retrieve_archived_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation from archive."""
        archive_file = self.archive_path / f"{session_id}.json.gz"
        
        if archive_file.exists():
            with gzip.open(archive_file, 'rt', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    
    def _rebuild_indices(self):
        """Rebuild search indices after pruning."""
        conversations = self._load_json(self.conversations_file)
        
        # Reset indices
        new_index = {
            "total_conversations": len(conversations),
            "topics": {},
            "keywords": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # Rebuild
        for session_id, conv in conversations.items():
            # Topics
            for topic in conv.get("topics", []):
                if topic not in new_index["topics"]:
                    new_index["topics"][topic] = []
                new_index["topics"][topic].append(session_id)
            
            # Keywords (simplified)
            for msg in conv["messages"][:10]:  # Only index first 10 messages for performance
                words = msg["content"].lower().split()
                keywords = [w for w in words if len(w) > 4]
                
                for keyword in keywords[:20]:  # Limit keywords per message
                    if keyword not in new_index["keywords"]:
                        new_index["keywords"][keyword] = []
                    if session_id not in new_index["keywords"][keyword]:
                        new_index["keywords"][keyword].append(session_id)
        
        self._save_json(self.memory_index_file, new_index)
    
    def configure_pruning(self, max_conversations: int = None, 
                         max_age_days: int = None,
                         max_storage_mb: int = None):
        """
        Configure pruning settings.
        
        Args:
            max_conversations: Maximum number of active conversations
            max_age_days: Maximum age of conversations in days
            max_storage_mb: Maximum storage size in MB
        """
        if max_conversations is not None:
            self.max_active_conversations = max_conversations
        if max_age_days is not None:
            self.max_message_age_days = max_age_days
        if max_storage_mb is not None:
            self.max_storage_mb = max_storage_mb
    def paste_conversation(self, pasted_text: str, user_id: str = "default",
                          title: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle pasted conversation text with automatic parsing and summarization.
        Designed for Android copy-paste workflow.
        
        Args:
            pasted_text: The pasted conversation text
            user_id: User identifier
            title: Optional title for the conversation
            
        Returns:
            Import result with session_id and summary
        """
        # Clean up common paste artifacts
        cleaned_text = self._clean_pasted_text(pasted_text)
        
        # Generate unique ID for this paste
        text_hash = hashlib.md5(cleaned_text.encode()).hexdigest()[:8]
        timestamp = datetime.datetime.now().isoformat()
        session_id = f"pasted_{user_id}_{text_hash}_{timestamp}"
        
        # Detect and parse format
        messages = self._parse_conversation_text(cleaned_text, "auto")
        
        if not messages:
            return {
                "success": False,
                "error": "Could not parse any messages from the pasted text"
            }
        
        # Generate automatic summary
        summary = self._generate_summary(messages)
        
        # Extract topics
        topics = self._extract_topics(messages)
        
        # Create conversation entry
        conversations = self._load_json(self.conversations_file)
        
        # Estimate conversation timespan
        first_timestamp = messages[0]["timestamp"]
        last_timestamp = messages[-1]["timestamp"]
        
        conversations[session_id] = {
            "user_id": user_id,
            "started_at": first_timestamp,
            "ended_at": last_timestamp,
            "messages": messages,
            "summary": summary,
            "topics": topics,
            "title": title or f"Conversation about {topics[0] if topics else 'various topics'}",
            "imported": True,
            "import_method": "paste",
            "message_count": len(messages)
        }
        
        self._save_json(self.conversations_file, conversations)
        
        # Update indices
        for msg in messages:
            self._update_keywords(session_id, msg["content"])
        
        # Update topic index
        if topics:
            self._update_topic_index(session_id, topics)
        
        return {
            "success": True,
            "session_id": session_id,
            "title": conversations[session_id]["title"],
            "summary": summary,
            "topics": topics,
            "message_count": len(messages),
            "user_messages": len([m for m in messages if m["role"] == "user"]),
            "amelia_messages": len([m for m in messages if m["role"] == "amelia"])
        }
    
    def _clean_pasted_text(self, text: str) -> str:
        """Clean up common artifacts from pasted text."""
        # Remove common copy-paste artifacts
        text = text.strip()
        
        # Fix common encoding issues
        replacements = {
            '"': '"', '"': '"', ''': "'", ''': "'",
            '…': '...', '–': '-', '—': '-'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove common UI elements that might be copied
        ui_patterns = [
            r'^\d+:\d+\s*[AP]M\s*$',  # Timestamps alone on lines
            r'^Sent from .+$',          # Email footers
            r'^\s*Copy\s*$',            # UI buttons
            r'^\s*Share\s*$',
            r'^\s*Reply\s*$'
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            skip = False
            for pattern in ui_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    skip = True
                    break
            if not skip:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _generate_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate an intelligent summary of the conversation.
        Uses NLP techniques without requiring external libraries.
        """
        if not messages:
            return "Empty conversation"
        
        # Extract key information
        total_messages = len(messages)
        user_messages = [m for m in messages if m["role"] == "user"]
        amelia_messages = [m for m in messages if m["role"] == "amelia"]
        
        # Analyze conversation flow
        all_text = ' '.join([m["content"] for m in messages])
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Remove common words (simple stop words)
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'what', 'how', 'when', 'where', 'who', 'why',
            'i', 'you', 'we', 'they', 'he', 'she', 'would', 'could', 'should',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'can', 'been',
            'being', 'be', 'are', 'was', 'were', 'am', 'very', 'really', 'just'
        }
        
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(meaningful_words)
        
        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(5)]
        
        # Identify question-answer patterns
        questions = []
        for msg in user_messages:
            if '?' in msg["content"]:
                questions.append(msg["content"])
        
        # Analyze conversation characteristics
        characteristics = []
        
        # Length characteristic
        if total_messages < 5:
            characteristics.append("brief")
        elif total_messages < 20:
            characteristics.append("moderate length")
        else:
            characteristics.append("extended")
        
        # Topic depth (based on keyword repetition)
        if len(word_freq) < total_messages * 2:
            characteristics.append("focused")
        else:
            characteristics.append("wide-ranging")
        
        # Interaction style
        if len(questions) > len(user_messages) * 0.5:
            characteristics.append("exploratory")
        elif len(questions) < len(user_messages) * 0.2:
            characteristics.append("informational")
        else:
            characteristics.append("conversational")
        
        # Build summary
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"A {', '.join(characteristics)} discussion")
        
        # Main topics
        if top_keywords:
            summary_parts.append(f"covering {', '.join(top_keywords[:3])}")
        
        # Question focus
        if questions:
            key_question_words = []
            for q in questions[:2]:
                q_words = re.findall(r'\b\w+\b', q.lower())
                important_words = [w for w in q_words if w not in stop_words and len(w) > 3]
                key_question_words.extend(important_words[:2])
            
            if key_question_words:
                unique_q_words = list(set(key_question_words) - set(top_keywords))
                if unique_q_words:
                    summary_parts.append(f"with questions about {', '.join(unique_q_words[:2])}")
        
        # Message count
        summary_parts.append(f"({total_messages} messages: {len(user_messages)} from user, {len(amelia_messages)} from Amelia)")
        
        return '. '.join(summary_parts) + '.'
    
    def _extract_topics(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        Extract main topics from the conversation using NLP techniques.
        """
        # Combine all message content
        all_text = ' '.join([m["content"] for m in messages])
        
        # Look for capitalized phrases (often topics)
        capitalized_phrases = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', all_text)
        
        # Look for quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', all_text) + re.findall(r"'([^']+)'", all_text)
        
        # Common topic indicators
        topic_patterns = [
            r'about\s+(\w+(?:\s+\w+){0,2})',
            r'regarding\s+(\w+(?:\s+\w+){0,2})',
            r'topic\s+of\s+(\w+(?:\s+\w+){0,2})',
            r'discuss(?:ing|ed)?\s+(\w+(?:\s+\w+){0,2})',
            r'question\s+about\s+(\w+(?:\s+\w+){0,2})',
            r'(\w+(?:\s+\w+){0,2})\s+is\s+(?:a|an|the)',
            r'what\s+is\s+(\w+(?:\s+\w+){0,2})'
        ]
        
        topic_candidates = []
        
        # Extract from patterns
        for pattern in topic_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            topic_candidates.extend([m.lower() for m in matches])
        
        # Add quoted terms and significant capitalized phrases
        topic_candidates.extend([t.lower() for t in quoted_terms if len(t) > 3])
        topic_candidates.extend([p.lower() for p in capitalized_phrases if len(p) > 5])
        
        # Get word frequencies for validation
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        
        # Filter and rank topics
        valid_topics = []
        seen = set()
        
        for topic in topic_candidates:
            topic_clean = topic.strip().lower()
            
            # Skip if already seen or too short
            if topic_clean in seen or len(topic_clean) < 4:
                continue
            
            # Check if topic words appear frequently enough
            topic_words = topic_clean.split()
            topic_score = sum(word_freq.get(w, 0) for w in topic_words)
            
            if topic_score >= 2:  # Mentioned at least twice
                valid_topics.append((topic_clean, topic_score))
                seen.add(topic_clean)
        
        # Sort by score and return top topics
        valid_topics.sort(key=lambda x: x[1], reverse=True)
        
        # Also add high-frequency meaningful words as topics
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but'}
        freq_words = [
            word for word, count in word_freq.most_common(10)
            if word not in stop_words and len(word) > 4 and count >= 3
        ]
        
        # Combine and deduplicate
        final_topics = []
        for topic, _ in valid_topics[:5]:
            final_topics.append(topic)
        
        for word in freq_words[:3]:
            if word not in ' '.join(final_topics):
                final_topics.append(word)
        
        return final_topics[:5]  # Return top 5 topics
    
    def paste_multiple_conversations(self, pasted_text: str, 
                                   user_id: str = "default") -> Dict[str, Any]:
        """
        Handle multiple conversations pasted at once.
        Automatically splits and processes each conversation.
        
        Args:
            pasted_text: The pasted text containing multiple conversations
            user_id: User identifier
            
        Returns:
            Import statistics and results
        """
        cleaned_text = self._clean_pasted_text(pasted_text)
        
        # Try to intelligently split conversations
        conversations = self._split_conversations(cleaned_text)
        
        results = {
            "total_attempted": len(conversations),
            "successful": 0,
            "failed": 0,
            "conversations": []
        }
        
        for i, conv_text in enumerate(conversations):
            if conv_text.strip():
                try:
                    result = self.paste_conversation(
                        conv_text, 
                        user_id,
                        title=f"Conversation {i+1}"
                    )
                    
                    if result["success"]:
                        results["successful"] += 1
                        results["conversations"].append(result)
                    else:
                        results["failed"] += 1
                        
                except Exception as e:
                    results["failed"] += 1
                    print(f"Failed to import conversation {i+1}: {e}")
        
        return results
    
    def _split_conversations(self, text: str) -> List[str]:
        """
        Intelligently split multiple conversations.
        """
        # Look for common separators
        separator_patterns = [
            r'\n={3,}\n',              # === separators
            r'\n-{3,}\n',              # --- separators
            r'\n\*{3,}\n',             # *** separators
            r'\n#{3,}\n',              # ### separators
            r'\nEND CONVERSATION\n',   # Explicit markers
            r'\nNEW CONVERSATION\n',
            r'\n\n\n\n+',              # Multiple blank lines
            r'\n\d{4}-\d{2}-\d{2}\n\n' # Date separators
        ]
        
        # Try each separator pattern
        for pattern in separator_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                conversations = re.split(pattern, text, flags=re.IGNORECASE)
                # If we found meaningful splits, use them
                if len(conversations) > 1:
                    return [c.strip() for c in conversations if c.strip()]
        
        # If no separators found, try to detect conversation breaks
        # by looking for large time gaps or greeting patterns
        lines = text.split('\n')
        conversations = []
        current_conv = []
        
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening)',
            r'^(goodbye|bye|see you|talk to you later)',
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this looks like a conversation start
            is_greeting = any(re.match(p, line, re.IGNORECASE) for p in greeting_patterns)
            
            # If we find a greeting after we've collected messages, 
            # it might be a new conversation
            if is_greeting and len(current_conv) > 5:
                conversations.append('\n'.join(current_conv))
                current_conv = [line]
            else:
                current_conv.append(line)
        
        # Don't forget the last conversation
        if current_conv:
            conversations.append('\n'.join(current_conv))
        
        # If we only found one conversation, return it as is
        if len(conversations) <= 1:
            return [text]
        
        return conversations


# Bridge functions for Chaquopy integration
def create_memory_module(storage_path: str = "amelia_memory") -> MemoryModule:
    """Create and return a memory module instance."""
    return MemoryModule(storage_path)

def start_new_conversation(memory: MemoryModule, user_id: str = "default") -> str:
    """Start a new conversation and return session ID."""
    return memory.start_conversation(user_id)

def add_user_message(memory: MemoryModule, content: str) -> bool:
    """Add a user message to current conversation."""
    return memory.add_message("user", content)

def add_amelia_message(memory: MemoryModule, content: str) -> bool:
    """Add an Amelia message to current conversation."""
    return memory.add_message("amelia", content)

def get_current_context(memory: MemoryModule, limit: int = 10) -> List[Dict[str, str]]:
    """Get recent conversation context."""
    return memory.get_conversation_context(message_limit=limit)

def search_memories(memory: MemoryModule, query: str) -> List[Dict[str, Any]]:
    """Search memories by keyword."""
    return memory.search_by_keyword(query)

def end_current_conversation(memory: MemoryModule, summary: str = None, topics: List[str] = None):
    """End the current conversation."""
    memory.end_conversation(summary, topics)

def import_past_conversation(memory: MemoryModule, text: str, user_id: str = "default", 
                            format_type: str = "auto") -> str:
    """Import a past conversation from text."""
    return memory.import_conversation_text(text, user_id, format_type=format_type)

def bulk_import_from_file(memory: MemoryModule, file_path: str) -> Dict[str, Any]:
    """Bulk import conversations from a file."""
    return memory.bulk_import_conversations(file_path)

def prune_old_memories(memory: MemoryModule, dry_run: bool = False) -> Dict[str, Any]:
    """Prune old conversations."""
    return memory.prune_old_conversations(dry_run)

def get_memory_stats(memory: MemoryModule) -> Dict[str, Any]:
    """Get memory statistics."""
    return memory.get_statistics()

# Android-friendly paste functions
def paste_single_conversation(memory: MemoryModule, pasted_text: str, 
                            user_id: str = "default", 
                            title: str = None) -> Dict[str, Any]:
    """
    Paste a single conversation - perfect for Android copy/paste.
    Returns summary and details about the imported conversation.
    """
    return memory.paste_conversation(pasted_text, user_id, title)

def paste_multiple_conversations(memory: MemoryModule, pasted_text: str,
                                user_id: str = "default") -> Dict[str, Any]:
    """
    Paste multiple conversations at once - handles bulk copy/paste.
    Automatically splits and processes each conversation.
    """
    return memory.paste_multiple_conversations(pasted_text, user_id)

def quick_paste(memory: MemoryModule, text: str) -> str:
    """
    Ultra-simple paste function for Android - just returns success message.
    """
    result = memory.paste_conversation(text)
    if result["success"]:
        return f"✓ Imported: {result['title']} ({result['message_count']} messages)"
    else:
        return "✗ Failed to import conversation"

# Android Asset functions
def import_from_asset(memory: MemoryModule, asset_content: str, 
                     asset_filename: str, user_id: str = "default") -> Dict[str, Any]:
    """
    Import conversation from Android asset file.
    Call this after reading the asset file content in Kotlin.
    """
    return memory.import_from_android_asset(asset_content, asset_filename, user_id)

def import_all_assets(memory: MemoryModule, asset_files: List[Tuple[str, str]], 
                     user_id: str = "default") -> Dict[str, Any]:
    """
    Import all conversation files from assets.
    asset_files should be a list of (filename, content) tuples.
    """
    return memory.import_all_from_assets(asset_files, user_id)
