import { Routes } from '@angular/router';
import { LiveView } from '../../pages/live_view/liveView';
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
        path: '**',
        redirectTo: 'live-view'
    }
];
