import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { usePushNotifications } from '@/hooks/usePushNotifications';
import { Bell, BellOff, Plus, Trash2, Clock, Pill, Heart } from 'lucide-react';
import { cn } from '@/lib/utils';

export function NotificationSettings() {
  const {
    permission,
    isSupported,
    reminders,
    requestPermission,
    addReminder,
    deleteReminder,
    toggleReminder,
    sendNotification,
  } = usePushNotifications();

  const [showAddForm, setShowAddForm] = useState(false);
  const [newReminder, setNewReminder] = useState({
    title: '',
    body: '',
    time: '09:00',
    type: 'health' as 'health' | 'medication',
  });

  const handleAddReminder = () => {
    if (!newReminder.title.trim()) return;
    
    addReminder({
      ...newReminder,
      enabled: true,
    });
    
    setNewReminder({ title: '', body: '', time: '09:00', type: 'health' });
    setShowAddForm(false);
  };

  const handleTestNotification = () => {
    sendNotification(
      'Test Notification',
      'Your notifications are working correctly!'
    );
  };

  if (!isSupported) {
    return (
      <Card className="border-destructive/30 bg-destructive/5">
        <CardContent className="flex items-center gap-3 p-4">
          <BellOff className="h-5 w-5 text-destructive" />
          <p className="text-sm text-muted-foreground">
            Push notifications are not supported in your browser.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="shadow-card border-border/50">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Bell className="h-5 w-5" />
              Health Reminders
            </CardTitle>
            <CardDescription>
              Set up daily reminders for health activities and medications
            </CardDescription>
          </div>
          {permission !== 'granted' && (
            <Button onClick={requestPermission} size="sm" className="gradient-health">
              Enable Notifications
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {permission === 'granted' && (
          <div className="flex items-center justify-between p-3 rounded-lg bg-success/10 border border-success/30">
            <div className="flex items-center gap-2">
              <Bell className="h-4 w-4 text-success" />
              <span className="text-sm text-success">Notifications enabled</span>
            </div>
            <Button variant="outline" size="sm" onClick={handleTestNotification}>
              Test
            </Button>
          </div>
        )}

        {permission === 'denied' && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30">
            <p className="text-sm text-destructive">
              Notifications are blocked. Please enable them in your browser settings.
            </p>
          </div>
        )}

        {/* Reminder List */}
        <div className="space-y-3">
          {reminders.map((reminder) => (
            <div
              key={reminder.id}
              className={cn(
                'flex items-center gap-4 p-3 rounded-lg border transition-colors',
                reminder.enabled ? 'bg-muted/30' : 'bg-muted/10 opacity-60'
              )}
            >
              <div className="flex-shrink-0">
                {reminder.type === 'medication' ? (
                  <Pill className="h-5 w-5 text-primary" />
                ) : (
                  <Heart className="h-5 w-5 text-health" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm truncate">{reminder.title}</p>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>{reminder.time}</span>
                </div>
              </div>
              <Switch
                checked={reminder.enabled}
                onCheckedChange={() => toggleReminder(reminder.id)}
                disabled={permission !== 'granted'}
              />
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground hover:text-destructive"
                onClick={() => deleteReminder(reminder.id)}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))}
        </div>

        {/* Add Reminder Form */}
        {showAddForm ? (
          <div className="space-y-4 p-4 rounded-lg border bg-muted/20">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="title">Reminder Title</Label>
                <Input
                  id="title"
                  placeholder="e.g., Take morning medication"
                  value={newReminder.title}
                  onChange={(e) => setNewReminder(prev => ({ ...prev, title: e.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="type">Type</Label>
                <Select
                  value={newReminder.type}
                  onValueChange={(value: 'health' | 'medication') => 
                    setNewReminder(prev => ({ ...prev, type: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="health">Health Reminder</SelectItem>
                    <SelectItem value="medication">Medication Alert</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="time">Time</Label>
                <Input
                  id="time"
                  type="time"
                  value={newReminder.time}
                  onChange={(e) => setNewReminder(prev => ({ ...prev, time: e.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="body">Description (optional)</Label>
                <Input
                  id="body"
                  placeholder="Additional details..."
                  value={newReminder.body}
                  onChange={(e) => setNewReminder(prev => ({ ...prev, body: e.target.value }))}
                />
              </div>
            </div>
            <div className="flex gap-2 justify-end">
              <Button variant="outline" onClick={() => setShowAddForm(false)}>
                Cancel
              </Button>
              <Button onClick={handleAddReminder} disabled={!newReminder.title.trim()}>
                Add Reminder
              </Button>
            </div>
          </div>
        ) : (
          <Button
            variant="outline"
            className="w-full"
            onClick={() => setShowAddForm(true)}
          >
            <Plus className="mr-2 h-4 w-4" />
            Add Custom Reminder
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
