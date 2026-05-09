import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-signup',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.css']
})
export class SignupComponent {
  email = '';
  password = '';
  confirmPassword = '';
  errorMessage = '';
  isLoading = false;
  showPassword = false;
  showConfirmPassword = false;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  onSubmit(): void {
    this.errorMessage = '';

    // Validation
    if (!this.email || !this.password || !this.confirmPassword) {
      this.errorMessage = 'Please fill in all fields';
      return;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(this.email)) {
      this.errorMessage = 'Please enter a valid email address (e.g., user@example.com)';
      return;
    }

    // Password length validation
    if (this.password.length < 8) {
      this.errorMessage = 'Password must be at least 8 characters long';
      return;
    }

    if (this.password.length > 72) {
      this.errorMessage = 'Password cannot be longer than 72 characters';
      return;
    }

    // Check for password complexity
    const hasUpperCase = /[A-Z]/.test(this.password);
    const hasLowerCase = /[a-z]/.test(this.password);
    const hasNumber = /[0-9]/.test(this.password);

    if (!hasUpperCase || !hasLowerCase || !hasNumber) {
      this.errorMessage = 'Password must contain at least one uppercase letter, one lowercase letter, and one number';
      return;
    }

    if (this.password !== this.confirmPassword) {
      this.errorMessage = 'Passwords do not match. Please check and try again.';
      return;
    }

    this.isLoading = true;

    this.authService.signup(this.email, this.password).subscribe({
      next: () => {
        this.isLoading = false;
        // Redirect to login page after successful signup
        this.router.navigate(['/login'], {
          queryParams: { message: 'Account created successfully! Please sign in.' }
        });
      },
      error: (error) => {
        this.isLoading = false;
        const detail = error.error?.detail;
        
        // Provide specific error messages
        if (detail && detail.includes('already registered')) {
          this.errorMessage = 'This email is already registered. Please login or use a different email.';
        } else if (detail) {
          this.errorMessage = detail;
        } else if (error.status === 0) {
          this.errorMessage = 'Cannot connect to server. Please make sure the backend is running.';
        } else {
          this.errorMessage = 'Signup failed. Please check your information and try again.';
        }
      }
    });
  }

  togglePasswordVisibility(): void {
    this.showPassword = !this.showPassword;
  }

  toggleConfirmPasswordVisibility(): void {
    this.showConfirmPassword = !this.showConfirmPassword;
  }
}

// Made with Bob
