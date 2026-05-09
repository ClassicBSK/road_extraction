import { Injectable, PLATFORM_ID, Inject } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, tap } from 'rxjs';
import { Router } from '@angular/router';

export interface User {
  id: number;
  email: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Project {
  id: number;
  name: string;
  owner_id: number;
  created_at: string;
  updated_at: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://localhost:8000';
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  public currentUser$ = this.currentUserSubject.asObservable();
  private isBrowser: boolean;

  constructor(
    private http: HttpClient,
    private router: Router,
    @Inject(PLATFORM_ID) platformId: Object
  ) {
    this.isBrowser = isPlatformBrowser(platformId);
    
    // Check if user is already logged in (only in browser)
    if (this.isBrowser) {
      const token = this.getToken();
      if (token) {
        this.loadCurrentUser();
      }
    }
  }

  private getToken(): string | null {
    if (!this.isBrowser) return null;
    return localStorage.getItem('access_token');
  }

  private setToken(token: string): void {
    if (!this.isBrowser) return;
    localStorage.setItem('access_token', token);
  }

  private removeToken(): void {
    if (!this.isBrowser) return;
    localStorage.removeItem('access_token');
  }

  private getAuthHeaders(): HttpHeaders {
    const token = this.getToken();
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });
  }

  signup(email: string, password: string): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`${this.apiUrl}/auth/signup`, {
      email,
      password
    });
    // Don't auto-login after signup - let user go to login page
  }

  login(email: string, password: string): Observable<AuthResponse> {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);

    return this.http.post<AuthResponse>(`${this.apiUrl}/auth/login`, formData).pipe(
      tap(response => {
        this.setToken(response.access_token);
        this.currentUserSubject.next(response.user);
      })
    );
  }

  logout(): void {
    this.removeToken();
    this.currentUserSubject.next(null);
    this.router.navigate(['/login']);
  }

  loadCurrentUser(): void {
    this.http.get<User>(`${this.apiUrl}/auth/me`, {
      headers: this.getAuthHeaders()
    }).subscribe({
      next: (user) => this.currentUserSubject.next(user),
      error: () => {
        this.removeToken();
        this.currentUserSubject.next(null);
      }
    });
  }

  isAuthenticated(): boolean {
    if (!this.isBrowser) return false;
    return !!this.getToken();
  }

  // Project methods
  getProjects(): Observable<Project[]> {
    return this.http.get<Project[]>(`${this.apiUrl}/projects/`, {
      headers: this.getAuthHeaders()
    });
  }

  createProject(name: string): Observable<Project> {
    return this.http.post<Project>(`${this.apiUrl}/projects/`, 
      { name },
      { headers: this.getAuthHeaders() }
    );
  }

  getProject(id: number): Observable<Project> {
    return this.http.get<Project>(`${this.apiUrl}/projects/${id}`, {
      headers: this.getAuthHeaders()
    });
  }
}

// Made with Bob
