import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService, Project, User } from '../../services/auth.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css']
})
export class DashboardComponent implements OnInit {
  currentUser: User | null = null;
  projects: Project[] = [];
  isLoading = true;
  errorMessage = '';
  
  // Create project modal
  showCreateModal = false;
  newProjectName = '';
  createError = '';
  isCreating = false;

  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.authService.currentUser$.subscribe(user => {
      this.currentUser = user;
    });
    
    this.loadProjects();
  }

  loadProjects(): void {
    this.isLoading = true;
    this.errorMessage = '';
    
    this.authService.getProjects().subscribe({
      next: (projects) => {
        this.projects = projects.sort((a, b) => 
          new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        );
        this.isLoading = false;
      },
      error: (error) => {
        this.errorMessage = 'Failed to load projects';
        this.isLoading = false;
        console.error('Error loading projects:', error);
      }
    });
  }

  openCreateModal(): void {
    this.showCreateModal = true;
    this.newProjectName = '';
    this.createError = '';
  }

  closeCreateModal(): void {
    this.showCreateModal = false;
    this.newProjectName = '';
    this.createError = '';
  }

  createProject(): void {
    if (!this.newProjectName.trim()) {
      this.createError = 'Project name is required';
      return;
    }

    // Check if project name already exists
    if (this.projects.some(p => p.name.toLowerCase() === this.newProjectName.trim().toLowerCase())) {
      this.createError = 'A project with this name already exists';
      return;
    }

    this.isCreating = true;
    this.createError = '';

    this.authService.createProject(this.newProjectName.trim()).subscribe({
      next: (project) => {
        this.isCreating = false;
        this.closeCreateModal();
        // Navigate to live view with the new project
        this.router.navigate(['/live-view'], { 
          queryParams: { projectId: project.id, projectName: project.name }
        });
      },
      error: (error) => {
        this.isCreating = false;
        this.createError = error.error?.detail || 'Failed to create project';
      }
    });
  }

  openProject(project: Project): void {
    this.router.navigate(['/live-view'], {
      queryParams: { projectId: project.id, projectName: project.name }
    });
  }

  goToLiveView(): void {
    // Navigate to live-view without a project (user can create one there)
    this.router.navigate(['/live-view']);
  }

  logout(): void {
    this.authService.logout();
  }

  formatDate(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
}

// Made with Bob
