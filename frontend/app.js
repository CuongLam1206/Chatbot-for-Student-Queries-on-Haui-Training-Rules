/**
 * Frontend JavaScript for Agentic RAG Web Application
 * Handles API communication, UI updates, and conversation management
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000/api';

// State
let currentSessionId = null;
let conversations = [];
let isLoading = false;
let deleteTargetSessionId = null;

// DOM Elements
const elements = {
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    messagesContainer: document.getElementById('messagesContainer'),
    conversationsList: document.getElementById('conversationsList'),
    newConversationBtn: document.getElementById('newConversationBtn'),
    sidebarToggle: document.getElementById('sidebarToggle'),
    sidebar: document.getElementById('sidebar'),
    deleteModal: document.getElementById('deleteModal'),
    confirmDeleteBtn: document.getElementById('confirmDeleteBtn'),
    cancelDeleteBtn: document.getElementById('cancelDeleteBtn'),
    charCounter: document.getElementById('charCounter'),
    statusIndicator: document.getElementById('statusIndicator')
};

// ============ Initialization ============

async function init() {
    console.log('🚀 Initializing Agentic RAG Web App');
    
    // Load session from localStorage or create new one
    currentSessionId = localStorage.getItem('currentSessionId');
    if (!currentSessionId) {
        await createNewConversation();
    }
    
    // Load conversations list
    await loadConversations();
    
    // Load current conversation messages
    if (currentSessionId) {
        await loadConversationMessages(currentSessionId);
    }
    
    // Check API health
    await checkHealth();
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('✅ App initialized');
}

// ============ API Functions ============

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateStatusIndicator(true, 'Connected');
        } else {
            updateStatusIndicator(false, 'Unhealthy');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatusIndicator(false, 'Disconnected');
    }
}

async function sendMessage(message) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
}

async function loadConversations() {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations`);
        const data = await response.json();
        conversations = data;
        renderConversationsList();
    } catch (error) {
        console.error('Error loading conversations:', error);
        showError('Failed to load conversations');
    }
}

async function loadConversationMessages(sessionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations/${sessionId}`);
        const data = await response.json();
        
        // Clear welcome message
        elements.messagesContainer.innerHTML = '';
        
        // Render messages
        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(msg => {
                addMessageToUI(msg.role, msg.content, false);
            });
            scrollToBottom();
        } else {
            showWelcomeMessage();
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
        showWelcomeMessage();
    }
}

async function createNewConversation() {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations/new`, {
            method: 'POST'
        });
        const data = await response.json();
        
        currentSessionId = data.session_id;
        localStorage.setItem('currentSessionId', currentSessionId);
        
        // Clear messages and show welcome
        elements.messagesContainer.innerHTML = '';
        showWelcomeMessage();
        
        // Reload conversations list
        await loadConversations();
        
        console.log('✅ New conversation created:', currentSessionId);
    } catch (error) {
        console.error('Error creating conversation:', error);
        showError('Failed to create new conversation');
    }
}

async function deleteConversation(sessionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/conversations/${sessionId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete conversation');
        }
        
        // If deleted conversation was current, create new one
        if (sessionId === currentSessionId) {
            await createNewConversation();
        }
        
        // Reload conversations list
        await loadConversations();
        
        console.log('✅ Conversation deleted:', sessionId);
    } catch (error) {
        console.error('Error deleting conversation:', error);
        showError('Failed to delete conversation');
    }
}

// ============ UI Functions ============

function setupEventListeners() {
    // Send message
    elements.sendBtn.addEventListener('click', handleSendMessage);
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    
    // Auto-resize textarea
    elements.messageInput.addEventListener('input', (e) => {
        e.target.style.height = 'auto';
        e.target.style.height = e.target.scrollHeight + 'px';
        
        // Update character counter
        const length = e.target.value.length;
        elements.charCounter.textContent = `${length} / 2000`;
    });
    
    // New conversation
    elements.newConversationBtn.addEventListener('click', createNewConversation);
    
    // Sidebar toggle
    elements.sidebarToggle.addEventListener('click', () => {
        elements.sidebar.classList.toggle('collapsed');
    });
    
    // Delete modal
    elements.confirmDeleteBtn.addEventListener('click', async () => {
        if (deleteTargetSessionId) {
            await deleteConversation(deleteTargetSessionId);
            closeDeleteModal();
        }
    });
    
    elements.cancelDeleteBtn.addEventListener('click', closeDeleteModal);
    
    // Close modal on outside click
    elements.deleteModal.addEventListener('click', (e) => {
        if (e.target === elements.deleteModal) {
            closeDeleteModal();
        }
    });
}

async function handleSendMessage() {
    const message = elements.messageInput.value.trim();
    
    if (!message || isLoading) return;
    
    // Add user message to UI
    addMessageToUI('user', message);
    
    // Clear input
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    elements.charCounter.textContent = '0 / 2000';
    
    // Show typing indicator
    showTypingIndicator();
    
    isLoading = true;
    elements.sendBtn.disabled = true;
    
    try {
        // Send to API
        const response = await sendMessage(message);
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add assistant response to UI
        addMessageToUI('assistant', response.answer);
        
        // Reload conversations list to update title if needed
        await loadConversations();
        
    } catch (error) {
        removeTypingIndicator();
        showError('Failed to send message. Please try again.');
        console.error(error);
    } finally {
        isLoading = false;
        elements.sendBtn.disabled = false;
        elements.messageInput.focus();
    }
}

function addMessageToUI(role, content, shouldScroll = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = content;
    
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(contentDiv);
    
    elements.messagesContainer.appendChild(messageDiv);
    
    if (shouldScroll) {
        scrollToBottom();
    }
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message message-assistant';
    typingDiv.id = 'typing-indicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const dotsDiv = document.createElement('div');
    dotsDiv.className = 'typing-indicator';
    dotsDiv.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
    
    contentDiv.appendChild(dotsDiv);
    typingDiv.appendChild(contentDiv);
    elements.messagesContainer.appendChild(typingDiv);
    
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function showWelcomeMessage() {
    elements.messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🎯</div>
            <h2>Chào mừng đến với Agentic RAG!</h2>
            <p>Tôi là trợ lý AI thông minh, có thể giúp bạn:</p>
            <ul>
                <li>🔍 Trả lời câu hỏi về quy chế đào tạo</li>
                <li>📋 Tìm kiếm thông tin nhanh chóng</li>
                <li>💡 Giải thích các điều khoản phức tạp</li>
                <li>📚 Tham khảo nguồn gốc thông tin</li>
            </ul>
            <p class="welcome-footer">Hãy bắt đầu bằng cách đặt câu hỏi bên dưới!</p>
        </div>
    `;
}

function renderConversationsList() {
    if (conversations.length === 0) {
        elements.conversationsList.innerHTML = `
            <div class="loading-conversations">
                <p>Chưa có cuộc hội thoại nào</p>
            </div>
        `;
        return;
    }
    
    elements.conversationsList.innerHTML = '';
    
    conversations.forEach(conv => {
        const convDiv = document.createElement('div');
        convDiv.className = 'conversation-item';
        if (conv.session_id === currentSessionId) {
            convDiv.classList.add('active');
        }
        
        const infoDiv = document.createElement('div');
        infoDiv.className = 'conversation-info';
        
        const titleDiv = document.createElement('div');
        titleDiv.className = 'conversation-title';
        titleDiv.textContent = conv.title || 'New Conversation';
        
        const metaDiv = document.createElement('div');
        metaDiv.className = 'conversation-meta';
        metaDiv.innerHTML = `
            <span>${formatDate(conv.updated_at)}</span>
            <span>${conv.message_count || 0} messages</span>
        `;
        
        infoDiv.appendChild(titleDiv);
        infoDiv.appendChild(metaDiv);
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-delete-conversation';
        deleteBtn.textContent = '×';
        deleteBtn.title = 'Delete conversation';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            openDeleteModal(conv.session_id);
        };
        
        convDiv.appendChild(infoDiv);
        convDiv.appendChild(deleteBtn);
        
        convDiv.onclick = () => switchConversation(conv.session_id);
        
        elements.conversationsList.appendChild(convDiv);
    });
}

async function switchConversation(sessionId) {
    if (sessionId === currentSessionId) return;
    
    currentSessionId = sessionId;
    localStorage.setItem('currentSessionId', sessionId);
    
    await loadConversationMessages(sessionId);
    renderConversationsList(); // Update active state
}

function openDeleteModal(sessionId) {
    deleteTargetSessionId = sessionId;
    elements.deleteModal.classList.add('active');
}

function closeDeleteModal() {
    deleteTargetSessionId = null;
    elements.deleteModal.classList.remove('active');
}

function updateStatusIndicator(isHealthy, text) {
    const dot = elements.statusIndicator.querySelector('.status-dot');
    const statusText = elements.statusIndicator.querySelector('.status-text');
    
    if (isHealthy) {
        dot.style.background = 'var(--success-color)';
        statusText.style.color = 'var(--success-color)';
    } else {
        dot.style.background = 'var(--error-color)';
        statusText.style.color = 'var(--error-color)';
    }
    
    statusText.textContent = text;
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message message-assistant';
    errorDiv.innerHTML = `
        <div class="message-content" style="background: rgba(239, 68, 68, 0.1); border-color: var(--error-color);">
            <div class="message-text" style="color: var(--error-color);">
                ⚠️ ${message}
            </div>
        </div>
    `;
    elements.messagesContainer.appendChild(errorDiv);
    scrollToBottom();
}

function scrollToBottom() {
    setTimeout(() => {
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
    }, 100);
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit' });
}

// ============ Start App ============

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
