type MessageHandler = (data: any) => void;

export class TaskWebSocket {
  private socket: WebSocket | null = null;
  private url: string;
  private onMessage: MessageHandler;

  constructor(url: string, onMessage: MessageHandler) {
    this.url = url;
    this.onMessage = onMessage;
  }

  connect() {
    this.socket = new WebSocket(this.url);
    this.socket.onopen = () => {
      console.log('WebSocket 连接已建立');
    };
    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.onMessage(data);
    };
    this.socket.onclose = (event) => {
      console.log('WebSocket 连接已关闭', event.code, event.reason);
    };
    this.socket.onerror = (error) => {
      console.error('WebSocket 错误:', error);
    };
  }

  send(data: any) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    }
  }

  close() {
    if (this.socket) {
      this.socket.close();
    }
  }
}