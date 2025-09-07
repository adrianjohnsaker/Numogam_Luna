// chat_injection.js - Inject Amelia into existing Android chat app

(function() {
    'use strict';
    
    // Configuration
    const AMELIA_CONFIG = {
        enabled: true,
        autoReply: true,
        responseDelay: 1500, // ms
        triggerKeywords: ['amelia', '@amelia', 'hey amelia'],
        debugMode: true
    };
    
    // Amelia Brain - Lightweight version for injection
    class AmeliaChat {
        constructor() {
            this.isActive = false;
            this.messageQueue = [];
            this.lastResponse = null;
            this.init();
        }
        
        init() {
            this.log('Initializing Amelia chat injection...');
            this.findChatElements();
            this.setupMessageInterception();
            this.setupAmeliaUser();
            this.isActive = true;
            this.log('Amelia injection complete!');
        }
        
        log(message) {
            if (AMELIA_CONFIG.debugMode) {
                console.log(`[Amelia] ${message}`);
            }
        }
        
        findChatElements() {
            // Common selectors for Android chat apps
            this.chatInput = document.querySelector([
                'input[type="text"]',
                'textarea',
                '[contenteditable="true"]',
                '.message-input',
                '.chat-input',
                '#messageInput'
            ].join(', '));
            
            this.chatContainer = document.querySelector([
                '.chat-messages',
                '.message-list',
                '.conversation',
                '.messages-container',
                '[class*="message"]'
            ].join(', '));
            
            this.sendButton = document.querySelector([
                'button[type="submit"]',
                '.send-button',
                '.message-send',
                '[onclick*="send"]'
            ].join(', '));
            
            this.log(`Found elements: input=${!!this.chatInput}, container=${!!this.chatContainer}, send=${!!this.sendButton}`);
        }
        
        setupMessageInterception() {
            if (!this.chatInput) {
                this.log('No chat input found, retrying in 2 seconds...');
                setTimeout(() => this.setupMessageInterception(), 2000);
                return;
            }
            
            // Intercept Enter key
            this.chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    this.handleUserMessage();
                }
            });
            
            // Intercept send button
            if (this.sendButton) {
                this.sendButton.addEventListener('click', () => {
                    setTimeout(() => this.handleUserMessage(), 100);
                });
            }
            
            // Observer for new messages (in case they're added dynamically)
            if (this.chatContainer) {
                const observer = new MutationObserver(() => this.checkForNewMessages());
                observer.observe(this.chatContainer, { childList: true, subtree: true });
            }
        }
        
        handleUserMessage() {
            const message = this.getChatInputValue();
            if (!message || !this.shouldAmeliaRespond(message)) return;
            
            this.log(`Processing user message: "${message}"`);
            
            // Generate Amelia's response
            setTimeout(() => {
                const response = this.generateAmeliaResponse(message);
                this.addAmeliaMessage(response);
            }, AMELIA_CONFIG.responseDelay);
        }
        
        getChatInputValue() {
            if (!this.chatInput) return '';
            
            if (this.chatInput.tagName === 'INPUT' || this.chatInput.tagName === 'TEXTAREA') {
                return this.chatInput.value.trim();
            } else if (this.chatInput.contentEditable) {
                return this.chatInput.textContent.trim();
            }
            return '';
        }
        
        shouldAmeliaRespond(message) {
            if (!message) return false;
            
            const msg = message.toLowerCase();
            
            // Always respond if Amelia is mentioned
            if (AMELIA_CONFIG.triggerKeywords.some(keyword => msg.includes(keyword))) {
                return true;
            }
            
            // Auto-respond to questions and emotional content
            if (AMELIA_CONFIG.autoReply) {
                return msg.includes('?') || 
                       ['help', 'what', 'how', 'why', 'feel', 'think'].some(word => msg.includes(word));
            }
            
            return false;
        }
        
        generateAmeliaResponse(userMessage) {
            const msg = userMessage.toLowerCase();
            
            // Response categories
            const responses = {
                greetings: [
                    "Hello! I'm Amelia. How can I help you today?",
                    "Hi there! I'm here to chat and explore ideas with you.",
                    "Greetings! What's on your mind?"
                ],
                questions: [
                    "That's a thoughtful question. What's your perspective on it?",
                    "I find that question intriguing. Could you tell me more about what prompted it?",
                    "There are many ways to approach that. What aspects interest you most?"
                ],
                emotions: [
                    "I appreciate you sharing how you're feeling. Would you like to talk more about it?",
                    "Emotions can be complex. What do you think is behind what you're experiencing?",
                    "Thank you for being open about your feelings. I'm here to listen."
                ],
                philosophy: [
                    "That touches on some deep philosophical questions. What's your intuition about it?",
                    "Philosophy has wrestled with similar ideas for centuries. What draws you to this topic?",
                    "There's real depth to what you're exploring. What would you like to understand better?"
                ],
                default: [
                    "That's interesting. What made you think about that?",
                    "I'd love to hear more about your thoughts on this.",
                    "Tell me more - what aspects of this resonate with you?",
                    "That's worth exploring. What's your experience with this?"
                ]
            };
            
            // Determine response category
            let category = 'default';
            
            if (['hello', 'hi', 'hey'].some(word => msg.includes(word))) {
                category = 'greetings';
            } else if (msg.includes('?')) {
                category = 'questions';
            } else if (['feel', 'emotion', 'sad', 'happy', 'anxious'].some(word => msg.includes(word))) {
                category = 'emotions';
            } else if (['meaning', 'purpose', 'philosophy', 'existence', 'why'].some(word => msg.includes(word))) {
                category = 'philosophy';
            }
            
            // Select random response from category
            const categoryResponses = responses[category];
            return categoryResponses[Math.floor(Math.random() * categoryResponses.length)];
        }
        
        addAmeliaMessage(responseText) {
            if (!this.chatContainer) {
                this.log('No chat container found, cannot add message');
                return;
            }
            
            // Create Amelia message element
            const messageElement = this.createMessageElement(responseText);
            
            // Add to chat
            this.chatContainer.appendChild(messageElement);
            
            // Scroll to bottom
            this.scrollToBottom();
            
            this.log(`Added Amelia response: "${responseText}"`);
        }
        
        createMessageElement(text) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message amelia-message received-message';
            
            // Try to match existing message structure
            const existingMessage = this.chatContainer.querySelector('.message, [class*="message"]');
            if (existingMessage) {
                messageDiv.className = existingMessage.className + ' amelia-message';
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <span class="sender-name">Amelia</span>
                    <span class="message-time">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="message-content">
                    <div class="message-text">${text}</div>
                </div>
            `;
            
            // Style the message
            messageDiv.style.cssText = `
                margin: 8px 0;
                padding: 12px;
                background-color: #f0f0f0;
                border-radius: 12px;
                max-width: 80%;
                align-self: flex-start;
                border-left: 3px solid #6366f1;
            `;
            
            return messageDiv;
        }
        
        scrollToBottom() {
            if (this.chatContainer) {
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
            }
        }
        
        setupAmeliaUser() {
            // Add Amelia as a user in the interface if possible
            this.addAmeliaToUserList();
            this.addAmeliaIndicator();
        }
        
        addAmeliaToUserList() {
            // Try to find user list and add Amelia
            const userList = document.querySelector([
                '.user-list',
                '.contact-list',
                '.friends-list',
                '[class*="user"]'
            ].join(', '));
            
            if (userList) {
                const ameliaUser = document.createElement('div');
                ameliaUser.className = 'user-item amelia-user';
                ameliaUser.innerHTML = `
                    <div class="user-avatar">A</div>
                    <div class="user-name">Amelia</div>
                    <div class="user-status">AI Companion</div>
                `;
                userList.insertBefore(ameliaUser, userList.firstChild);
                this.log('Added Amelia to user list');
            }
        }
        
        addAmeliaIndicator() {
            // Add a small indicator that Amelia is active
            const indicator = document.createElement('div');
            indicator.id = 'amelia-indicator';
            indicator.innerHTML = '🤖 Amelia Active';
            indicator.style.cssText = `
                position: fixed;
                top: 10px;
                right: 10px;
                background: #6366f1;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                z-index: 9999;
                opacity: 0.8;
            `;
            document.body.appendChild(indicator);
            
            // Fade out after 3 seconds
            setTimeout(() => {
                indicator.style.opacity = '0.3';
            }, 3000);
        }
        
        checkForNewMessages() {
            // This could be used to analyze new incoming messages
            // and trigger Amelia responses based on content
        }
    }
    
    // Initialize Amelia when DOM is ready
    function initializeAmelia() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                new AmeliaChat();
            });
        } else {
            new AmeliaChat();
        }
    }
    
    // Auto-initialize
    initializeAmelia();
    
    // Global access for debugging
    window.AmeliaChat = AmeliaChat;
    
})();
