package cmd

import (
	"encoding/base64"
	"fmt"
	"testing"
	"time"
)

// A metadata-server token with less than 45 minutes left must not be cached past its expiry.
func TestTokenCacheTTL(t *testing.T) {
	cases := []struct {
		lifetime, want time.Duration
	}{
		{3599 * time.Second, 45 * time.Minute}, // fresh token: capped at 45 min
		{20 * time.Minute, 18 * time.Minute},   // metadata cached token: expiry - 2 min margin
		{90 * time.Second, 0},                  // about to expire: do not cache
		{0, 5 * time.Minute},                   // unknown lifetime (gcloud): cache briefly
	}
	for _, c := range cases {
		if got := tokenCacheTTL(c.lifetime); got != c.want {
			t.Errorf("tokenCacheTTL(%v) = %v, want %v", c.lifetime, got, c.want)
		}
	}
}

func TestJWTLifetime(t *testing.T) {
	exp := time.Now().Add(10 * time.Minute).Unix()
	payload := base64.RawURLEncoding.EncodeToString([]byte(fmt.Sprintf(`{"exp":%d}`, exp)))
	got := jwtLifetime("h." + payload + ".s")
	if got < 9*time.Minute || got > 10*time.Minute {
		t.Errorf("jwtLifetime = %v, want ~10m", got)
	}
	if jwtLifetime("not-a-jwt") != 0 {
		t.Error("jwtLifetime of a non-JWT should be 0")
	}
}
