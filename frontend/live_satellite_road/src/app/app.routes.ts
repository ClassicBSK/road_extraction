import { Routes } from '@angular/router';
import { LiveView } from '../../pages/live_view/liveView';
<<<<<<< HEAD
import { LoginComponent } from './components/login/login.component';
import { SignupComponent } from './components/signup/signup.component';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
    {
        path: '',
        redirectTo: '/login',
        pathMatch: 'full'
    },
    {
        path: 'login',
        component: LoginComponent
    },
    {
        path: 'signup',
        component: SignupComponent
    },
    {
        path: 'dashboard',
        component: DashboardComponent,
        canActivate: [authGuard]
    },
    {
        path: 'live-view',
        component: LiveView,
        canActivate: [authGuard]
=======
import { PhotoUpload } from '../../pages/photo_upload/photoUpload';
export const routes: Routes = [
    {
        path: '',
        redirectTo: 'live-view',
        pathMatch: 'full'
    },
    {
        path: 'live-view',
        component: LiveView
    },
    {
        path: 'photo-upload',
        component: PhotoUpload
    },
    {
        path: '**',
        redirectTo: 'live-view'
>>>>>>> 6880dd06821d1d0597bd7de18ce51356bdd1d18a
    }
];

// Made with Bob
